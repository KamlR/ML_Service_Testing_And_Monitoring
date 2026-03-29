import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from ml_service.metrics.metrics import (
    HTTP_REQUEST_DURATION_SECONDS,
    HTTP_REQUEST_EXCEPTIONS_TOTAL, 
    HTTP_REQUESTS_TOTAL,
    update_runtime_metrics,
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        method = request.method
        path = request.url.path
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as exc:
            status_code = 500
            HTTP_REQUEST_EXCEPTIONS_TOTAL.labels(
                method=method,
                path=path,
                exception_type=exc.__class__.__name__,
            ).inc()
            raise
        finally:
            duration = time.perf_counter() - start_time
            HTTP_REQUESTS_TOTAL.labels(
                method=method,
                path=path,
                status_code=str(status_code),
            ).inc()
            HTTP_REQUEST_DURATION_SECONDS.labels(
                method=method,
                path=path,
            ).observe(duration)
            update_runtime_metrics()