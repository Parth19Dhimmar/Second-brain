import uuid
from fastapi.responses import JSONResponse


def generate_request_id():
    return f"req_{uuid.uuid4().hex[:10]}"


def success_response(data=None, message="Success", status_code=200):
    return JSONResponse(
        status_code=status_code,
        content={
            "success": True,
            "status_code": status_code,
            "data": data,
            "error": None,
            "message": message,
            "request_id": generate_request_id(),
        },
    )


def error_response(message, status_code=500, error_code="SERVER_ERROR"):
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "status_code": status_code,
            "data": None,
            "error": {
                "code": error_code,
                "message": message,
            },
            "message": "Request failed",
            "request_id": generate_request_id(),
        },
    )
