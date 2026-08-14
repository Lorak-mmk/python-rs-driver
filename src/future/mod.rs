use std::future::Future;
use std::sync::{Arc, Condvar, Mutex};
use std::task::Wake;
use crate::future::asyncio::waker::AsyncioWaker;
use crate::future::asyncio::{BoxedFuture, Coroutine, PollResult};
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyStopIteration;
use pyo3::exceptions::PyTimeoutError;
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::{BoundObject, Py, PyAny, PyResult};
mod asyncio;
mod callbacks;
/// Internal state of a PyDriverFuture.
enum FutureState {
    /// Future is driven by the asyncio executor.
    PendingAsyncio { coroutine: Coroutine },
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
