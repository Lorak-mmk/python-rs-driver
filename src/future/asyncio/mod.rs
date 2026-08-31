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
//
// - Upstream's `future: Option<BoxedFuture>` (emptied by `close()`) is replaced by an
//   unconditionally owned `BoxedFuture`: a `Coroutine` value always has a future to poll.
//   Every operation that gets rid of the future consumes the whole `Coroutine` instead of
//   emptying it — `poll` takes `self` and hands the coroutine back only in `PollResult::Pending`,
//   `into_future_and_waker` extracts the future so it can be spawned on tokio, and
//   `into_waker` drops the future on close. A spent coroutine is therefore not
//   representable, so upstream's "poll after completion" check is gone: the compiler
//   rules that case out.
//
// - `into_future_and_waker` returns an `Arc<AsyncioWaker>` that is shared between the
//   coroutine's previous parking and the tokio task, so a Python coroutine already
//   suspended on this future is woken by the spawned task.
//
// - `into_waker` drops the future and returns the waker so that the caller
//   (`PyDriverFuture::close_future`) can fire `waker.wake()` after writing `Ready`, ensuring any
//   Python coroutine suspended on this future gets rescheduled and sees the closed state.
//
// - Removed `unsafe impl Sync for Coroutine`. It is no longer needed because `Coroutine`
//   is not a `#[pyclass]` and lives behind a `Mutex`.

use std::future::Future;
use std::panic;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use crate::future::asyncio::waker::AsyncioWaker;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

pub(crate) mod waker;

pub(crate) type BoxedFuture = Pin<Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>>;

/// Result of polling a coroutine. In contrast to the Rust `Poll` enum, the pending
/// variant carries the value to yield to the Python event loop — and the coroutine
/// itself, since [`Coroutine::poll`] consumes it.
pub(crate) enum PollResult {
    /// The future is not ready, so the coroutine is handed back to be polled again.
    ///
    /// `value` is what should be yielded to the Python event loop: either an
    /// `asyncio.Future` object or `py.None()`. It is a `PyResult` because creating
    /// the `asyncio.Future` can fail (e.g. when there is no running event loop);
    /// the coroutine is still returned in that case, so the caller can keep it and
    /// just propagate the error.
    Pending {
        coroutine: Coroutine,
        value: PyResult<Py<PyAny>>,
    },
    /// The future completed with this result. The coroutine is consumed.
    Ready(PyResult<Py<PyAny>>),
}

/// Rust-side coroutine wrapping a [`Future`].
///
/// The future is owned unconditionally: as long as a `Coroutine` value exists, it has
/// a future to poll. Getting rid of the future means getting rid of the `Coroutine`.
pub(crate) struct Coroutine {
    future: BoxedFuture,
    waker: Option<Arc<AsyncioWaker>>,
}

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
            future: Box::pin(future),
            waker: None,
        }
    }

    /// Consume the coroutine, returning the inner future so it can be driven elsewhere
    /// (i.e. spawned on tokio), together with the waker it was parked on.
    ///
    /// The waker is the one an already-suspended Python coroutine is waiting on, so
    /// whoever drives the future from now on must wake it through this very waker.
    pub(crate) fn into_future_and_waker(self) -> (BoxedFuture, Arc<AsyncioWaker>) {
        let Coroutine { future, waker } = self;
        (
            future,
            waker.unwrap_or_else(|| Arc::new(AsyncioWaker::new())),
        )
    }

    /// Consume the coroutine, dropping the inner future, and return the waker it was
    /// parked on, if any.
    ///
    /// Used on close: the caller fires `waker.wake()` after writing the terminal state,
    /// so a Python coroutine suspended on this future gets rescheduled and observes it.
    pub(crate) fn into_waker(self) -> Option<Arc<AsyncioWaker>> {
        self.waker
    }

    /// Return the waker to poll with: reset in place when we hold the only reference,
    /// replaced by a fresh one when the event loop still holds the previous one.
    fn poll_waker(&mut self) -> Arc<AsyncioWaker> {
        if let Some(waker) = self.waker.as_mut() {
            match Arc::get_mut(waker) {
                Some(unique) => unique.reset(),
                None => *waker = Arc::new(AsyncioWaker::new()),
            }
        }
        Arc::clone(
            self.waker
                .get_or_insert_with(|| Arc::new(AsyncioWaker::new())),
        )
    }

    /// Poll the underlying future, consuming the coroutine.
    ///
    /// The coroutine is handed back in [`PollResult::Pending`] and only there, so a
    /// completed (or thrown-into, or panicked) future cannot be polled again.
    pub(crate) fn poll(mut self, py: Python<'_>, throw: Option<Py<PyAny>>) -> PollResult {
        // reraise thrown exception, dropping the future
        if let Some(exc) = throw {
            return PollResult::Ready(Err(PyErr::from_value(exc.into_bound(py))));
        }
        let asyncio_waker = self.poll_waker();
        let waker = Waker::from(Arc::clone(&asyncio_waker));
        // poll the Rust future and forward its results if ready
        // polling is UnwindSafe because the future is dropped in case of panic
        let future = &mut self.future;
        let poll = || future.as_mut().poll(&mut Context::from_waker(&waker));
        match panic::catch_unwind(panic::AssertUnwindSafe(poll)) {
            Ok(Poll::Ready(res)) => return PollResult::Ready(res),
            Ok(Poll::Pending) => {}
            Err(err) => {
                let msg = if let Some(s) = err.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = err.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "Rust future panicked".to_string()
                };
                return PollResult::Ready(Err(PyRuntimeError::new_err(msg)));
            }
        }

        let value = asyncio_waker.yield_asyncio_future(py);
        PollResult::Pending {
            coroutine: self,
            value,
        }
    }
}
