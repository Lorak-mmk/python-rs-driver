use crate::RUNTIME;
use crate::errors::FutureCancelledError;
use crate::future::asyncio::waker::AsyncioWaker;
use crate::future::asyncio::{BoxedFuture, Coroutine, PollResult};
use crate::future::callbacks::CallbackKind;
use crate::utils::PyDuration;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyStopIteration;
use pyo3::exceptions::PyTimeoutError;
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::{BoundObject, Py, PyAny, PyResult};
use std::future::Future;
use std::sync::{Arc, Condvar, Mutex};
use std::task::Wake;
use std::time::Duration;

use tokio::task::AbortHandle;

mod asyncio;
mod callbacks;

// # PyDriverFuture — hybrid design
//
// ## Three states
//
// `PendingAsyncio { coroutine }`
//     The future is driven by the asyncio event loop.
//     This is the default starting state.
//
// `PendingTokio { on_success, on_error, abort_handle, waker }`
//     The future has been spawned on the tokio runtime. `__next__` just
//     yields the asyncio future from the waker. The spawned task transitions
//     to `Ready` on completion.
//
// `Ready { result }`
//     Terminal state. Result stored permanently.
//
// ## Transitions
//
// - `PendingAsyncio` → `PendingTokio`: when callbacks are registered, `result()` is
//   called, or `start()` is called explicitly. The inner future is taken from the
//   coroutine, spawned on tokio.
// - `PendingAsyncio` → `Ready`: when `poll` completes, or `close()`/`cancel()` is called.
// - `PendingTokio` → `Ready`: when the spawned task completes, or `close()`/`cancel()` aborts it.
// - `Ready` → (no transitions)

/// Internal state of a PyDriverFuture.
enum FutureState {
    /// Future is driven by the asyncio executor.
    PendingAsyncio { coroutine: Coroutine },
    /// Future has been spawned on the tokio runtime.
    PendingTokio {
        callbacks: Vec<CallbackKind>,
        abort_handle: Option<AbortHandle>,
        waker: Arc<AsyncioWaker>,
    },
    /// Future has completed. Result is stored permanently.
    Ready { result: PyResult<Py<PyAny>> },
}

struct FutureInner {
    state: Mutex<FutureState>,
    /// Notified when state transitions to Ready.
    ready: Condvar,
}

/// A Python awaitable wrapping a Rust future.
#[pyclass(name = "DriverFuture", frozen)]
pub struct PyDriverFuture {
    inner: Arc<FutureInner>,
}

impl PyDriverFuture {
    /// Create a PyDriverFuture starting in PendingAsyncio (default).
    fn new<F>(future: F) -> Self
    where
        F: Future<Output = PyResult<Py<PyAny>>> + Send + 'static,
    {
        Self {
            inner: Arc::new(FutureInner {
                state: Mutex::new(FutureState::PendingAsyncio {
                    coroutine: Coroutine::new(future),
                }),
                ready: Condvar::new(),
            }),
        }
    }

    /// Create a `Py<PyDriverFuture>` from a future returning `Result<T, E>`.
    /// Starts in PendingAsyncio.
    pub(crate) fn spawn<Fut, T, E>(py: Python<'_>, future: Fut) -> PyResult<Py<PyDriverFuture>>
    where
        Fut: Future<Output = Result<T, E>> + Send + 'static,
        T: for<'py> IntoPyObject<'py>,
        E: Into<PyErr>,
    {
        Py::new(
            py,
            PyDriverFuture::new(async move {
                let result = future.await;
                Python::attach(|py| {
                    result.map_err(Into::into).and_then(|v| {
                        v.into_pyobject(py)
                            .map(|b| b.into_any().unbind())
                            .map_err(Into::into)
                    })
                })
            }),
        )
    }
    /// Create an already-resolved PyDriverFuture.
    pub(crate) fn ready(py: Python, result: PyResult<Py<PyAny>>) -> PyResult<Py<PyDriverFuture>> {
        Py::new(
            py,
            PyDriverFuture {
                inner: Arc::new(FutureInner {
                    state: Mutex::new(FutureState::Ready { result }),
                    ready: Condvar::new(),
                }),
            },
        )
    }

    /// Spawn a future on tokio, returning the abort handle.
    /// On completion the spawned task transitions `state` to `Ready`,
    /// fires any registered callbacks, wakes the asyncio waker, and notifies
    /// the condvar.
    fn spawn_future_on_tokio<F>(
        future: F,
        inner: &Arc<FutureInner>,
        waker: &Arc<AsyncioWaker>,
    ) -> AbortHandle
    where
        F: Future<Output = PyResult<Py<PyAny>>> + Send + 'static,
    {
        let inner_clone = Arc::clone(inner);
        let waker_clone = Arc::clone(waker);

        let handle = RUNTIME.spawn(async move {
            let result = future.await;

            Python::attach(|py| {
                let callbacks = {
                    let mut state = inner_clone.state.lock_py_attached(py).unwrap();
                    match &mut *state {
                        FutureState::PendingTokio { callbacks, .. } => {
                            let taken = std::mem::take(callbacks);
                            *state = FutureState::Ready {
                                result: clone_result(py, &result),
                            };
                            Some(taken)
                        }
                        _ => None,
                    }
                };

                // `None` means the future was already closed/cancelled/thrown-into
                // by the time this task completed. There is nothing left to notify.
                let Some(callbacks) = callbacks else {
                    return;
                };

                if callbacks.is_empty() {
                    waker_clone.wake();
                    inner_clone.ready.notify_all();
                    return;
                }

                let result_for_cbs = clone_result(py, &result);
                RUNTIME.spawn_blocking(move || {
                    Python::attach(|py| {
                        CallbackKind::fire_all(py, callbacks, &result_for_cbs);
                    });

                    waker_clone.wake();
                    inner_clone.ready.notify_all();
                });
            });
        });

        handle.abort_handle()
    }

    /// Transition from PendingAsyncio to PendingTokio by spawning the given
    /// future on the tokio runtime.
    /// Must be called while holding the state lock.
    fn transition_to_tokio(
        future: BoxedFuture,
        waker: Arc<AsyncioWaker>,
        inner: &Arc<FutureInner>,
        state_guard: &mut std::sync::MutexGuard<'_, FutureState>,
    ) {
        let abort_handle = Self::spawn_future_on_tokio(future, inner, &waker);

        **state_guard = FutureState::PendingTokio {
            callbacks: Vec::new(),
            abort_handle: Some(abort_handle),
            waker,
        };
    }

    /// If `state_guard` is `PendingAsyncio`, take its future/waker and
    /// transition to `PendingTokio`. No-op otherwise.
    /// Must be called while holding the state lock.
    fn ensure_started(
        inner: &Arc<FutureInner>,
        state_guard: &mut std::sync::MutexGuard<'_, FutureState>,
    ) {
        if let FutureState::PendingAsyncio { coroutine } = &mut **state_guard {
            let (future, waker) = coroutine
                .take_future_and_waker()
                .expect("PendingAsyncio coroutine has no future");
            Self::transition_to_tokio(future, waker, inner, state_guard);
        }
    }

    /// Poll the coroutine (__next__).
    fn poll_coroutine(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mut state = self.inner.state.lock_py_attached(py).unwrap();
        match &mut *state {
            FutureState::Ready { result } => Err(raise_stop_iteration(py, result)),

            FutureState::PendingTokio { waker, .. } => {
                // Future is running on tokio — just yield the asyncio future.
                let waker = Arc::clone(waker);
                drop(state);
                waker.yield_asyncio_future(py)
            }

            FutureState::PendingAsyncio { coroutine } => {
                // Drive the future via the coroutine.
                match coroutine.poll(py, None)? {
                    PollResult::Pending(maybe_future) => Ok(maybe_future),
                    PollResult::Ready(result) => {
                        *state = FutureState::Ready {
                            result: clone_result(py, &result),
                        };
                        drop(state);
                        self.inner.ready.notify_all();
                        Err(raise_stop_iteration(py, &result))
                    }
                }
            }
        }
    }

    /// Close the future. Transitions to Ready with `exc` as the error.
    fn close_future(&self, py: Python<'_>, exc: PyErr) {
        let err_result: PyResult<Py<PyAny>> = Err(exc);

        let (callbacks, waker) = {
            let mut state = self.inner.state.lock_py_attached(py).unwrap();

            let (callbacks, waker) = match &mut *state {
                FutureState::Ready { .. } => return,

                FutureState::PendingTokio {
                    abort_handle,
                    waker,
                    callbacks,
                    ..
                } => {
                    if let Some(ah) = abort_handle {
                        ah.abort();
                    }
                    (Some(std::mem::take(callbacks)), Some(Arc::clone(waker)))
                }

                FutureState::PendingAsyncio { coroutine } => {
                    (None, coroutine.close_and_get_waker())
                }
            };

            *state = FutureState::Ready {
                result: clone_result(py, &err_result),
            };

            (callbacks, waker)
        };

        self.inner.ready.notify_all();

        if let Some(waker) = waker {
            waker.wake();
        }

        if let Some(callbacks) = callbacks {
            CallbackKind::fire_all(py, callbacks, &err_result);
        }
    }

    /// Release the GIL, wait on the condvar until state is Ready or `timeout`
    /// elapses, then return the result. Raises `TimeoutError` on timeout.
    fn wait_for_ready(&self, py: Python<'_>, timeout: Option<Duration>) -> PyResult<Py<PyAny>> {
        let timed_out = py.detach(|| {
            let state = self.inner.state.lock().unwrap();
            match timeout {
                None => {
                    let _guard = self
                        .inner
                        .ready
                        .wait_while(state, |s| !matches!(s, FutureState::Ready { .. }))
                        .unwrap();
                    false
                }
                Some(timeout) => {
                    let (guard, result) = self
                        .inner
                        .ready
                        .wait_timeout_while(state, timeout, |s| {
                            !matches!(s, FutureState::Ready { .. })
                        })
                        .unwrap();

                    result.timed_out() && !matches!(*guard, FutureState::Ready { .. })
                }
            }
        });

        if timed_out {
            return Err(PyTimeoutError::new_err("DriverFuture.result() timed out"));
        }

        let state = self.inner.state.lock_py_attached(py).unwrap();
        match &*state {
            FutureState::Ready { result } => clone_result(py, result),
            _ => unreachable!("condvar woke but state is not Ready"),
        }
    }

    /// Block until the future is ready, returning the result.
    /// If `timeout` elapses first, raises `TimeoutError`.
    fn block_until_ready(&self, py: Python<'_>, timeout: Option<Duration>) -> PyResult<Py<PyAny>> {
        let mut state = self.inner.state.lock_py_attached(py).unwrap();
        match &mut *state {
            FutureState::Ready { result } => clone_result(py, result),

            FutureState::PendingTokio { .. } => {
                drop(state);
                self.wait_for_ready(py, timeout)
            }

            FutureState::PendingAsyncio { .. } => {
                Self::ensure_started(&self.inner, &mut state);
                drop(state);
                self.wait_for_ready(py, timeout)
            }
        }
    }

    /// Register a [`CallbackKind`] on this future.
    ///
    /// - If already `Ready`, invokes the callback immediately.
    /// - If `PendingTokio`, queues it.
    /// - If `PendingAsyncio`, transitions to `PendingTokio` first, then queues it.
    fn register_callback(&self, py: Python<'_>, cb: CallbackKind) {
        let mut state = self.inner.state.lock_py_attached(py).unwrap();
        match &mut *state {
            FutureState::Ready { result } => {
                let result = clone_result(py, result);
                drop(state);
                cb.invoke(py, &result);
            }

            FutureState::PendingTokio { callbacks, .. } => {
                callbacks.push(cb);
            }

            FutureState::PendingAsyncio { .. } => {
                Self::ensure_started(&self.inner, &mut state);
                if let FutureState::PendingTokio { callbacks, .. } = &mut *state {
                    callbacks.push(cb);
                }
            }
        }
    }

    /// Throw an exception into the future.
    /// - Ready: re-raises the exception (coroutine is exhausted).
    /// - PendingAsyncio: delegates to `coroutine.poll(py, Some(exc))`.
    /// - PendingTokio: aborts the tokio task, fires on_error callbacks,
    ///   transitions to Ready, and re-raises the exception.
    fn throw_into(&self, py: Python<'_>, exc: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let mut state = self.inner.state.lock_py_attached(py).unwrap();
        match &mut *state {
            FutureState::Ready { .. } => Err(PyErr::from_value(exc.into_bound(py))),

            FutureState::PendingAsyncio { coroutine } => match coroutine.poll(py, Some(exc))? {
                PollResult::Pending(value) => Ok(value),
                PollResult::Ready(result) => {
                    *state = FutureState::Ready {
                        result: clone_result(py, &result),
                    };
                    drop(state);
                    self.inner.ready.notify_all();
                    Err(raise_stop_iteration(py, &result))
                }
            },

            FutureState::PendingTokio {
                abort_handle,
                waker,
                callbacks,
                ..
            } => {
                if let Some(ah) = abort_handle {
                    ah.abort();
                }
                let waker = Arc::clone(waker);
                let taken = std::mem::take(callbacks);
                let err_result: PyResult<Py<PyAny>> =
                    Err(PyErr::from_value(exc.clone_ref(py).into_bound(py)));
                *state = FutureState::Ready {
                    result: clone_result(py, &err_result),
                };
                drop(state);

                waker.wake();
                self.inner.ready.notify_all();
                CallbackKind::fire_all(py, taken, &err_result);

                // Re-raise the thrown exception.
                err_result
            }
        }
    }
}

fn clone_result(py: Python<'_>, result: &PyResult<Py<PyAny>>) -> PyResult<Py<PyAny>> {
    match result {
        Ok(value) => Ok(value.clone_ref(py)),
        Err(err) => Err(err.clone_ref(py)),
    }
}

fn raise_stop_iteration(py: Python<'_>, result: &PyResult<Py<PyAny>>) -> PyErr {
    match result {
        Ok(value) => PyStopIteration::new_err((value.clone_ref(py),)),
        Err(err) => err.clone_ref(py),
    }
}

#[pymethods]
impl PyDriverFuture {
    fn __await__(self_: Py<Self>) -> Py<Self> {
        self_
    }

    fn __iter__(self_: Py<Self>) -> Py<Self> {
        self_
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.poll_coroutine(py)
    }

    fn send(&self, py: Python<'_>, _value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.__next__(py)
    }

    fn throw(&self, py: Python<'_>, exc: Py<PyAny>) -> PyResult<Py<PyAny>> {
        self.throw_into(py, exc)
    }

    fn close(&self, py: Python<'_>) {
        self.close_future(py, PyRuntimeError::new_err("future was closed"));
    }

    /// Cancel the future. Unlike `close()`, this raises `FutureCancelledError`
    /// from `result()`/`__next__()`/callbacks, distinguishing a deliberate
    /// cancellation from the future being torn down.
    fn cancel(&self, py: Python<'_>) {
        self.close_future(py, FutureCancelledError::new_err("future was cancelled"));
    }

    /// Get the result of this future.
    ///
    /// If the future is still pending, this blocks the calling thread until
    /// it completes (releasing the GIL while waiting). If `timeout` is
    /// given and elapses before the future completes, raises `TimeoutError`.
    #[pyo3(signature = (timeout=None))]
    fn result(&self, py: Python<'_>, timeout: Option<PyDuration>) -> PyResult<Py<PyAny>> {
        self.block_until_ready(py, timeout.map(|d| d.0))
    }

    /// Force the transition from `PendingAsyncio` to `PendingTokio`.
    ///
    /// Spawns the inner future onto the tokio runtime immediately, without
    /// waiting for a callback registration or a `result()` call. No-op if
    /// the future is already `PendingTokio` or `Ready`. Returns `self` so
    /// calls can be chained, e.g. `future = session.execute(...).start()`.
    fn start(self_: Py<Self>, py: Python<'_>) -> Py<Self> {
        {
            let this = self_.borrow(py);
            let mut state = this.inner.state.lock_py_attached(py).unwrap();
            Self::ensure_started(&this.inner, &mut state);
        }
        self_
    }

    /// Register a callback to be invoked when the future completes successfully.
    ///
    /// The callback is called as `callback(result)`.
    /// If the future is already done with a success, the callback is invoked immediately.
    /// If the future is pending on asyncio, it is moved to tokio to support callbacks.
    fn on_result(&self, py: Python<'_>, callback: Py<PyAny>) {
        let cb = CallbackKind::on_success(callback);
        self.register_callback(py, cb);
    }

    /// Register a callback to be invoked when the future completes with an error.
    ///
    /// The callback is called as `callback(exception)`.
    /// If the future is already done with an error, the callback is invoked immediately.
    /// If the future is pending on asyncio, it is moved to tokio to support callbacks.
    fn on_error(&self, py: Python<'_>, callback: Py<PyAny>) {
        let cb = CallbackKind::on_error(callback);
        self.register_callback(py, cb);
    }

    /// Returns True if the future has completed (successfully or with an error).
    fn done(&self, py: Python<'_>) -> bool {
        let state = self.inner.state.lock_py_attached(py).unwrap();
        matches!(*state, FutureState::Ready { .. })
    }

    /// Returns True if the future completed because `cancel()` was called.
    fn cancelled(&self, py: Python<'_>) -> bool {
        let state = self.inner.state.lock_py_attached(py).unwrap();
        match &*state {
            FutureState::Ready { result: Err(err) } => {
                err.is_instance_of::<FutureCancelledError>(py)
            }
            _ => false,
        }
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let state = self.inner.state.lock_py_attached(py).unwrap();
        match &*state {
            FutureState::PendingAsyncio { .. } | FutureState::PendingTokio { .. } => {
                "<DriverFuture pending>".to_string()
            }
            FutureState::Ready { result } => match result {
                Ok(_) => "<DriverFuture finished>".to_string(),
                Err(e) => format!("<DriverFuture finished exception={}>", e),
            },
        }
    }
}

#[pymodule]
pub(crate) fn future(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDriverFuture>()?;
    Ok(())
}
