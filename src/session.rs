use std::sync::Arc;

use crate::deserialize::results::RequestResult;
use crate::serialize::value_list::PyAnyWrapperValueList;
use crate::statement::PreparedStatement;
use pyo3::BoundObject;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use scylla::statement;
use scylla::statement::unprepared;

use crate::RUNTIME;
use crate::statement::Statement;
use pyo3::types::PyString;

#[pyclass]
pub(crate) struct Session {
    pub(crate) _inner: Arc<scylla::client::session::Session>,
}

#[pymethods]
impl Session {
    #[pyo3(signature = (request, values = Default::default()), text_signature = "(request, values = None)")]
    async fn execute(
        &self,
        request: ExecutableStatement,
        values: PyAnyWrapperValueList,
    ) -> PyResult<RequestResult> {
        let result = self
            .spawn_on_runtime(async move |s| {
                match request {
                    ExecutableStatement::Unprepared(statement) => {
                        s.query_unpaged(statement, values).await
                    }
                    ExecutableStatement::Prepared(statement) => {
                        s.execute_unpaged(&statement, values).await
                    }
                }
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })
            .await?;
        Ok(RequestResult {
            inner: Arc::new(result),
        })
    }

    async fn prepare(&self, statement: ExecutableStatement) -> PyResult<PreparedStatement> {
        match statement {
            ExecutableStatement::Unprepared(s) => self.scylla_prepare(s).await,
            ExecutableStatement::Prepared(prepared) => Ok(PreparedStatement { _inner: prepared }),
        }
    }
}

use pin_project::pin_project;

#[pin_project]
struct WithRuntime<Fut> {
    #[pin]
    f: Fut,
}

impl<Fut, R> Future for WithRuntime<Fut>
where
    Fut: Future<Output = PyResult<R>> + Send + 'static,
    R: Send + 'static,
{
    type Output = PyResult<R>;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let guard = RUNTIME.enter();
        let this = self.project();
        let result = Future::poll(this.f, cx);
        std::mem::drop(guard);
        result
    }
}

impl Session {
    async fn spawn_on_runtime<F, Fut, R>(&self, f: F) -> PyResult<R>
    where
        // closure: takes Arc<scylla::client::session::Session> and returns a future
        F: FnOnce(Arc<scylla::client::session::Session>) -> Fut + Send + 'static,
        // for spawn we need Send + 'static
        Fut: Future<Output = PyResult<R>> + Send + 'static,
        R: Send + 'static,
    {
        let session_clone = Arc::clone(&self._inner);

        let fut_with_runtime = WithRuntime {
            f: f(session_clone),
        };
        fut_with_runtime.await
    }

    async fn scylla_prepare(
        &self,
        statement: impl Into<statement::Statement>,
    ) -> PyResult<PreparedStatement> {
        match self._inner.prepare(statement).await {
            Ok(prepared) => Ok(PreparedStatement { _inner: prepared }),
            Err(e) => Err(PyErr::new::<PyRuntimeError, _>(format!(
                "Failed to prepare statement: {}",
                e
            ))),
        }
    }
}

#[derive(Clone)]
pub(crate) enum ExecutableStatement {
    Prepared(statement::prepared::PreparedStatement),
    Unprepared(unprepared::Statement),
}

impl<'a, 'py> FromPyObject<'a, 'py> for ExecutableStatement {
    type Error = PyErr;

    fn extract(request: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(prepared) = request.extract::<Py<PreparedStatement>>() {
            return Ok(ExecutableStatement::Prepared(prepared.get()._inner.clone()));
        }

        if let Ok(text) = request.extract::<Py<PyString>>() {
            return Ok(ExecutableStatement::Unprepared(
                text.to_str(request.py())?.into(),
            ));
        }

        if let Ok(statement) = request.extract::<Py<Statement>>() {
            return Ok(ExecutableStatement::Unprepared(
                statement.get()._inner.clone(),
            ));
        }

        Err(PyErr::new::<PyTypeError, _>(format!(
            "Invalid request type: expected str | Statement | PreparedStatement, got {}",
            request.into_bound().get_type().name()?
        )))
    }
}

#[pymodule]
pub(crate) fn session(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Session>()?;

    Ok(())
}
