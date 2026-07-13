// Portions of this file were copied from the PyO3 project (https://github.com/PyO3/pyo3),
// version 0.28.x (git commit: 8fcf8fc63), licensed under either of Apache-2.0 or MIT at your option.
//
// Copyright (c) 2023-present PyO3 Project and Contributors. https://github.com/PyO3
//
// Modifications Copyright 2025 ScyllaDB, licensed under Apache-2.0 OR MIT.
//
// Changes from the original pyo3 source:
// - Removed `ThrowCallback` and the `throw_callback` field from `Coroutine`.
//   In upstream pyo3, `ThrowCallback` is used to deliver exceptions thrown into
//   the coroutine to a `CancelHandle` (the `#[pyo3(cancel_handle)]` annotation).
//   Since we don't use `CancelHandle` in this project, the throw callback is
//   unnecessary. Now, `throw()` always drops the future and reraises the
//   exception directly (the simple path that upstream uses when no callback is set).
//
// - `Coroutine` is no longer a `#[pyclass]`. It is used purely as internal Rust state,
//   not exposed to Python directly. `poll` returns a `PollResult` enum (`Pending` / `Ready`)
//   instead of a Python object, keeping the result in the Rust type system. This avoids
//   the overhead and error-prone nature of converting to Python objects before the caller
//   is ready to use them, and allows building higher-level abstractions on top using
//   full Rust type guarantees.
//
// - Imports updated from pyo3-internal paths (`alloc`, `core`, `pyo3_macros`, `crate::platform`)
//   to standard `std` and public `pyo3::` re-exports, since this code lives outside the pyo3
//   crate itself.

use std::future::Future;
use std::panic;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use crate::future::asyncio::waker::AsyncioWaker;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyIterator;

pub(crate) mod waker;

type BoxedFuture = Pin<Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>>;

const COROUTINE_REUSED_ERROR: &str = "cannot reuse already awaited coroutine";

/// Result of polling a coroutine. In contrast to Rust Poll enum it stores Py<PyAny>
/// in Pending variant.
pub enum PollResult {
    /// The future is not ready. Yield this value to the Python event loop.
    /// Contains either an asyncio.Future object or py.None().
    Pending(Py<PyAny>),
    /// The future completed with this result.
    Ready(PyResult<Py<PyAny>>),
}

/// Rust-side coroutine wrapping a [`Future`].
pub(crate) struct Coroutine {
    future: Option<BoxedFuture>,
    waker: Option<Arc<AsyncioWaker>>,
}

// Safety: `Coroutine` is allowed to be `Sync` even though the future is not,
// because the future is polled with `&mut self` receiver
unsafe impl Sync for Coroutine {}

impl Coroutine {
    ///  Wrap a future into a Python coroutine.
    ///
    /// Coroutine `send` polls the wrapped future, ignoring the value passed
    /// (should always be `None` anyway).
    ///
    /// `Coroutine `throw` drop the wrapped future and reraise the exception passed
    pub(crate) fn new<F>(future: F) -> Self
    where
        F: Future<Output = PyResult<Py<PyAny>>> + Send + 'static,
    {
        Self {
            future: Some(Box::pin(future)),
            waker: None,
        }
    }

    /// Take the inner future out of this coroutine for use with block_on.
    /// After this, the coroutine is closed (poll will return an error).
    pub(crate) fn take_future(&mut self) -> Option<BoxedFuture> {
        self.future.take()
    }

    /// Poll the underlying future.
    pub(crate) fn poll(
        &mut self,
        py: Python<'_>,
        throw: Option<Py<PyAny>>,
    ) -> PyResult<PollResult> {
        // raise if the coroutine has already been run to completion
        let future_rs = match self.future {
            Some(ref mut fut) => fut,
            None => {
                return Ok(PollResult::Ready(Err(PyRuntimeError::new_err(
                    COROUTINE_REUSED_ERROR,
                ))));
            }
        };
        // reraise thrown exception
        if let Some(exc) = throw {
            self.close();
            return Ok(PollResult::Ready(Err(PyErr::from_value(
                exc.into_bound(py),
            ))));
        }
        // create a new waker, or try to reset it in place
        if let Some(waker) = self.waker.as_mut().and_then(Arc::get_mut) {
            waker.reset();
        } else {
            self.waker = Some(Arc::new(AsyncioWaker::new()));
        }
        let waker = Waker::from(self.waker.clone().unwrap());
        // poll the Rust future and forward its results if ready
        // polling is UnwindSafe because the future is dropped in case of panic
        let poll = || future_rs.as_mut().poll(&mut Context::from_waker(&waker));
        match std::panic::catch_unwind(panic::AssertUnwindSafe(poll)) {
            Ok(Poll::Ready(res)) => {
                self.close();
                return Ok(PollResult::Ready(res));
            }
            Err(err) => {
                self.close();
                let msg = if let Some(s) = err.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = err.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "Rust future panicked".to_string()
                };
                return Ok(PollResult::Ready(Err(PyRuntimeError::new_err(msg))));
            }
            _ => {}
        }

        // otherwise, initialize the waker `asyncio.Future`
        if let Some(future) = self.waker.as_ref().unwrap().initialize_future(py)? {
            // `asyncio.Future` must be awaited; fortunately, it implements `__iter__ = __await__`
            // and will yield itself if its result has not been set in polling above
            if let Some(future) = PyIterator::from_object(future).unwrap().next() {
                // future has not been leaked into Python for now, and Rust code can only call
                // `set_result(None)` in `Wake` implementation, so it's safe to unwrap
                return Ok(PollResult::Pending(future.unwrap().unbind()));
            }
        }
        // if waker has been woken during future polling, yield None (sleep(0) equivalent)
        Ok(PollResult::Pending(py.None()))
    }

    /// Close the coroutine, dropping the underlying future.
    pub(crate) fn close(&mut self) {
        drop(self.future.take());
    }
}
