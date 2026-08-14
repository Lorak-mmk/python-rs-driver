use std::future::Future;
use std::sync::{Arc, Condvar, Mutex};
use std::task::Wake;
use crate::future::asyncio::waker::AsyncioWaker;
use crate::future::asyncio::{BoxedFuture, Coroutine, PollResult};
use crate::future::callbacks::CallbackKind;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyStopIteration;
use pyo3::exceptions::PyTimeoutError;
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::{BoundObject, Py, PyAny, PyResult};

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
}

/// A Python awaitable wrapping a Rust future.
#[pyclass(name = "DriverFuture", frozen)]
pub struct PyDriverFuture {
    inner: Arc<FutureInner>,
}

impl PyDriverFuture {
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
        if let Some(waker) = waker {
            waker.wake();
        }

        if let Some(callbacks) = callbacks {
            CallbackKind::fire_all(py, callbacks, &err_result);
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
}

#[pymodule]
pub(crate) fn future(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDriverFuture>()?;
    Ok(())
}
