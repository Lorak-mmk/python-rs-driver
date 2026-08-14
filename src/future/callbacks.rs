use pyo3::prelude::*;
use pyo3::{Py, PyAny, PyResult};

/// A registered callback, invoked with the future's outcome as its sole argument.
pub(super) struct Callback {
    callable: Py<PyAny>,
}

impl Callback {
    fn new(callable: Py<PyAny>) -> Self {
        Self { callable }
    }

    /// Invoke this callback with `value` as its only argument.
    /// Errors are logged and swallowed.
    fn invoke(&self, py: Python<'_>, value: &Py<PyAny>) {
        if let Err(err) = self.callable.call1(py, (value.clone_ref(py),)) {
            log::error!("DriverFuture callback raised an exception: {}", err);
        }
    }
}

/// Discriminates whether a [`Callback`] fires on success or on error.
pub(super) enum CallbackKind {
    /// Fired when the future resolves successfully. Passes the result value.
    OnSuccess(Callback),
    /// Fired when the future resolves with an error. Passes the exception instance.
    OnError(Callback),
}

impl CallbackKind {
    pub(super) fn on_success(callable: Py<PyAny>) -> Self {
        Self::OnSuccess(Callback::new(callable))
    }

    pub(super) fn on_error(callable: Py<PyAny>) -> Self {
        Self::OnError(Callback::new(callable))
    }

    /// Invoke this callback if its variant matches the outcome of `result`.
    pub(super) fn invoke(&self, py: Python<'_>, result: &PyResult<Py<PyAny>>) {
        match (self, result) {
            (CallbackKind::OnSuccess(cb), Ok(value)) => {
                cb.invoke(py, value);
            }
            (CallbackKind::OnError(cb), Err(err)) => {
                let exc_obj = err.value(py);
                cb.invoke(py, exc_obj.as_any().as_unbound());
            }
            _ => {}
        }
    }

    /// Fire every callback in `callbacks` that matches the outcome of `result`.
    pub(super) fn fire_all(
        py: Python<'_>,
        callbacks: Vec<CallbackKind>,
        result: &PyResult<Py<PyAny>>,
    ) {
        for cb in &callbacks {
            cb.invoke(py, result);
        }
    }
}
