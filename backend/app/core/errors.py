"""Consistent JSON error responses. Stack traces are never returned to clients."""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("modeldoctor")


class ApiError(Exception):
    """Application error carrying an HTTP status code and a client-safe message."""

    def __init__(self, status_code: int, detail: str, code: str = "error"):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.code = code


def not_found(detail: str) -> ApiError:
    return ApiError(404, detail, "not_found")


def bad_request(detail: str) -> ApiError:
    return ApiError(400, detail, "bad_request")


def conflict(detail: str) -> ApiError:
    return ApiError(409, detail, "conflict")


def unprocessable(detail: str) -> ApiError:
    return ApiError(422, detail, "validation_error")


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail, "code": exc.code})

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc.detail), "code": "http_error"})

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        messages = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", []) if p != "body")
            messages.append(f"{loc}: {err.get('msg')}" if loc else str(err.get("msg")))
        return JSONResponse(
            status_code=422,
            content={"detail": "; ".join(messages) or "Invalid request", "code": "validation_error"},
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        logger.exception("Unhandled server error")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. Check the server logs for details.", "code": "internal_error"},
        )
