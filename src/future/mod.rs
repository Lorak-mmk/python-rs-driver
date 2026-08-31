use crate::RUNTIME;
use crate::errors::FutureCancelledError;
use crate::future::asyncio::waker::AsyncioWaker;
use crate::future::asyncio::{Coroutine, PollResult};
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
// ## States
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
// `Panicked`
//     Terminal state. A panic unwound out of a state transition, taking the coroutine
//     (or the tokio task) with it; every entry point reports `panicked_err()`.
//
// ## Transitions
//
// - `PendingAsyncio` → `PendingTokio`: when callbacks are registered, `result()` is
//   called, or `start()` is called explicitly. The coroutine is consumed and the future
//   it owns is spawned on tokio.
// - `PendingAsyncio` → `Ready`: when `poll` completes, or `close()`/`cancel()` is called.
// - `PendingTokio` → `Ready`: when the spawned task completes, or `close()`/`cancel()` aborts it.
// - any state → `Panicked`: when a panic unwinds out of a transition (see below).
// - `Ready` / `Panicked` → (no transitions)
//
// ## Ownership
//
// A `Coroutine` owns its future unconditionally — there is no "coroutine without a
// future" state to assert against. Consequently every operation that gets rid of the
// future (polling it to completion, spawning it on tokio, closing it) needs to *own*
// the coroutine, and the coroutine is owned by the state. So transitions take the whole
// state out of the mutex with `mem::replace`, match on it by value, and write the new
// state back before releasing the lock. `Panicked` is what sits in the mutex in between:
// a completed transition always overwrites it, so it survives only when a panic unwinds
// through one — which is exactly what it means.

/// Internal state of a PyDriverFuture.
enum FutureState {
    /// Future is driven by the asyncio executor.
    PendingAsyncio { coroutine: Coroutine },
    /// Future has been spawned on the tokio runtime.
    PendingTokio {
        callbacks: Vec<CallbackKind>,
        abort_handle: AbortHandle,
        waker: Arc<AsyncioWaker>,
    },
    /// Future has completed. Result is stored permanently.
    Ready { result: PyResult<Py<PyAny>> },
    /// A transition that consumes the previous state is in progress, or panicked
    /// halfway through one.
    ///
    /// `mem::replace`ing this in is how a transition takes ownership of the state it
    /// consumes; every transition writes a real state back before releasing the lock,
    /// so this is only ever observed by another thread if a panic unwound out of a
    /// transition. Then it is terminal: the coroutine and the tokio task, if any, are
    /// gone, so the future can no longer make progress.
    Panicked,
}

impl FutureState {
    /// Whether the future can still make progress. Both terminal states —
    /// [`FutureState::Ready`] and [`FutureState::Panicked`] — release
    /// [`FutureInner::ready`] waiters.
    fn is_terminal(&self) -> bool {
        matches!(self, FutureState::Ready { .. } | FutureState::Panicked)
    }
}

/// The error every entry point reports for a future left [`FutureState::Panicked`].
fn panicked_err() -> PyErr {
    PyRuntimeError::new_err("DriverFuture was left in an inconsistent state by a panic")
}

struct FutureInner {
    state: Mutex<FutureState>,
    /// Notified when the state transitions to a terminal state.
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
    /// Create a `Py<PyDriverFuture>` from a future returning `Result<T, E>`,
    /// spawning it on the tokio runtime immediately: the future starts in
    /// `PendingTokio` rather than lazily transitioning from `PendingAsyncio`.
    pub(crate) fn spawn_on_tokio<Fut, T, E>(
        py: Python<'_>,
        future: Fut,
    ) -> PyResult<Py<PyDriverFuture>>
    where
        Fut: Future<Output = Result<T, E>> + Send + 'static,
        T: for<'py> IntoPyObject<'py>,
        E: Into<PyErr>,
    {
        let wrapped = async move {
            let result = future.await;
            Python::attach(|py| {
                result.map_err(Into::into).and_then(|v| {
                    v.into_pyobject(py)
                        .map(|b| b.into_any().unbind())
                        .map_err(Into::into)
                })
            })
        };

        let waker = Arc::new(AsyncioWaker::new());
        let inner = Arc::new(FutureInner {
            state: Mutex::new(FutureState::Panicked),
            ready: Condvar::new(),
        });

        {
            // The initial `Panicked` is a placeholder: the task is spawned and the real
            // state written before the lock is released, and the spawned task cannot
            // observe the state without that lock.
            let mut state = inner.state.lock_py_attached(py).unwrap();
            let abort_handle = Self::spawn_future_on_tokio(wrapped, &inner, &waker);
            *state = FutureState::PendingTokio {
                callbacks: Vec::new(),
                abort_handle,
                waker,
            };
        }

        Py::new(py, PyDriverFuture { inner })
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

    /// If `state_guard` is `PendingAsyncio`, take its coroutine, spawn the future it
    /// owns on the tokio runtime, and transition to `PendingTokio`. No-op otherwise.
    /// Must be called while holding the state lock.
    fn ensure_started(
        inner: &Arc<FutureInner>,
        state_guard: &mut std::sync::MutexGuard<'_, FutureState>,
    ) {
        let coroutine = match std::mem::replace(&mut **state_guard, FutureState::Panicked) {
            FutureState::PendingAsyncio { coroutine } => coroutine,
            // Already started (or finished) — put the state back untouched.
            other => {
                **state_guard = other;
                return;
            }
        };

        let (future, waker) = coroutine.into_future_and_waker();
        let abort_handle = Self::spawn_future_on_tokio(future, inner, &waker);

        **state_guard = FutureState::PendingTokio {
            callbacks: Vec::new(),
            abort_handle,
            waker,
        };
    }

    /// Poll the coroutine (__next__).
    fn poll_coroutine(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mut state = self.inner.state.lock_py_attached(py).unwrap();
        match std::mem::replace(&mut *state, FutureState::Panicked) {
            FutureState::Ready { result } => {
                let err = raise_stop_iteration(py, &result);
                *state = FutureState::Ready { result };
                Err(err)
            }

            // Future is running on tokio — just yield the asyncio future.
            FutureState::PendingTokio {
                callbacks,
                abort_handle,
                waker,
            } => {
                let asyncio_waker = Arc::clone(&waker);
                *state = FutureState::PendingTokio {
                    callbacks,
                    abort_handle,
                    waker,
                };
                drop(state);
                asyncio_waker.yield_asyncio_future(py)
            }

            // Drive the future via the coroutine.
            FutureState::PendingAsyncio { coroutine } => match coroutine.poll(py, None) {
                PollResult::Pending { coroutine, value } => {
                    *state = FutureState::PendingAsyncio { coroutine };
                    value
                }
                PollResult::Ready(result) => {
                    *state = FutureState::Ready {
                        result: clone_result(py, &result),
                    };
                    drop(state);
                    self.inner.ready.notify_all();
                    Err(raise_stop_iteration(py, &result))
                }
            },

            // Nothing is left to drive the future — report the panic to the awaiter.
            FutureState::Panicked => Err(panicked_err()),
        }
    }

    /// Close the future. Transitions to Ready with `exc` as the error.
    fn close_future(&self, py: Python<'_>, exc: PyErr) {
        let err_result: PyResult<Py<PyAny>> = Err(exc);

        let (callbacks, waker) = {
            let mut state = self.inner.state.lock_py_attached(py).unwrap();

            // The state is replaced with its terminal value; whatever it held before
            // (the coroutine's future, the tokio task) is torn down here.
            let closed = FutureState::Ready {
                result: clone_result(py, &err_result),
            };
            match std::mem::replace(&mut *state, closed) {
                FutureState::Ready { result } => {
                    *state = FutureState::Ready { result };
                    return;
                }

                FutureState::PendingTokio {
                    callbacks,
                    abort_handle,
                    waker,
                } => {
                    abort_handle.abort();
                    (Some(callbacks), Some(waker))
                }

                FutureState::PendingAsyncio { coroutine } => (None, coroutine.into_waker()),

                // Nothing left to close; keep reporting the panic.
                FutureState::Panicked => {
                    *state = FutureState::Panicked;
                    return;
                }
            }
        };

        self.inner.ready.notify_all();

        if let Some(waker) = waker {
            waker.wake();
        }

        if let Some(callbacks) = callbacks {
            CallbackKind::fire_all(py, callbacks, &err_result);
        }
    }

    /// Release the GIL, wait on the condvar until the state is terminal or `timeout`
    /// elapses, then return the result. Raises `TimeoutError` on timeout.
    fn wait_for_ready(&self, py: Python<'_>, timeout: Option<Duration>) -> PyResult<Py<PyAny>> {
        let timed_out = py.detach(|| {
            let state = self.inner.state.lock().unwrap();
            match timeout {
                None => {
                    let _guard = self
                        .inner
                        .ready
                        .wait_while(state, |s| !s.is_terminal())
                        .unwrap();
                    false
                }
                Some(timeout) => {
                    let (guard, result) = self
                        .inner
                        .ready
                        .wait_timeout_while(state, timeout, |s| !s.is_terminal())
                        .unwrap();

                    result.timed_out() && !guard.is_terminal()
                }
            }
        });

        if timed_out {
            return Err(PyTimeoutError::new_err("DriverFuture.result() timed out"));
        }

        let state = self.inner.state.lock_py_attached(py).unwrap();
        match &*state {
            FutureState::Ready { result } => clone_result(py, result),
            // The condvar only lets us out on a terminal state, and the only other
            // terminal state is `Panicked`.
            _ => Err(panicked_err()),
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

            // Terminal, but there is no result to hand out.
            FutureState::Panicked => Err(panicked_err()),
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

            // The future will never complete, so a queued callback would never fire:
            // report the panic to `on_error` right away instead.
            FutureState::Panicked => {
                drop(state);
                cb.invoke(py, &Err(panicked_err()));
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
        match std::mem::replace(&mut *state, FutureState::Panicked) {
            FutureState::Ready { result } => {
                *state = FutureState::Ready { result };
                Err(PyErr::from_value(exc.into_bound(py)))
            }

            FutureState::PendingAsyncio { coroutine } => match coroutine.poll(py, Some(exc)) {
                PollResult::Pending { coroutine, value } => {
                    *state = FutureState::PendingAsyncio { coroutine };
                    value
                }
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
                callbacks,
                abort_handle,
                waker,
            } => {
                abort_handle.abort();
                let err_result: PyResult<Py<PyAny>> = Err(PyErr::from_value(exc.into_bound(py)));
                *state = FutureState::Ready {
                    result: clone_result(py, &err_result),
                };
                drop(state);

                waker.wake();
                self.inner.ready.notify_all();
                CallbackKind::fire_all(py, callbacks, &err_result);

                // Re-raise the thrown exception.
                err_result
            }

            // There is no coroutine left to throw into.
            FutureState::Panicked => {
                *state = FutureState::Panicked;
                Err(panicked_err())
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
        state.is_terminal()
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
            FutureState::Panicked => "<DriverFuture panicked>".to_string(),
        }
    }
}

#[pymodule]
pub(crate) fn future(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDriverFuture>()?;
    Ok(())
}
