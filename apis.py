"""
This module provides a FastAPI application that acts as a proxy for chat completion and embedding requests,
forwarding them to an underlying service running on a local port.
"""

import os
import logging
import httpx
import time
import uvicorn
from typing import Any
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import sys
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

LLM_URL = os.getenv("LLM_URL")
MODEL_ID = os.getenv("MODEL_ID")
FALL_BACK_URL = os.getenv("FALL_BACK_URL")
FALL_BACK_MODEL_ID = os.getenv("FALL_BACK_MODEL_ID")
API_KEY = os.getenv("API_KEY")

# Import schemas from schema.py
from schema import (
    ChatCompletionRequest
)

class ErrorHandlingStreamHandler(logging.StreamHandler):
    """Custom stream handler that handles I/O errors gracefully"""
    def emit(self, record):
        try:
            super().emit(record)
        except OSError as e:
            if e.errno == 5:  # Input/output error
                pass
            else:
                raise

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING if os.getenv("ENV") == "production" else logging.INFO)

handler = ErrorHandlingStreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Remove any existing handlers to avoid duplicate logging
for existing_handler in logger.handlers[:]:
    if not isinstance(existing_handler, ErrorHandlingStreamHandler):
        logger.removeHandler(existing_handler)

# Configure uvicorn access logger
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.handlers = []
uvicorn_access_handler = ErrorHandlingStreamHandler(sys.stderr)
uvicorn_access_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
uvicorn_access_logger.addHandler(uvicorn_access_handler)

# Constants
HTTP_TIMEOUT = 60.0
STREAM_TIMEOUT = 600.0
HEALTH_CHECK_INTERVAL = 30.0
MAX_CONNECTIONS = 100
MAX_KEEPALIVE_CONNECTIONS = 20
KEEPALIVE_TIMEOUT = 5.0
RATE_LIMIT_REQUESTS = 100  # requests per minute
RATE_LIMIT_WINDOW = 60  # seconds

# Global state for instance health
instance_state = {
    "last_health_check": 0,
    "is_primary_healthy": True,
    "current_url": LLM_URL,
    "current_model": MODEL_ID
}

# Rate limiting state
rate_limit_state = defaultdict(lambda: {"count": 0, "window_start": time.time()})

async def check_instance_health(url: str) -> bool:
    """Check if an instance is healthy"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{url}/health")
            return response.status_code == 200
    except Exception:
        return False

async def get_healthy_instance() -> tuple[str, str]:
    """Get a healthy instance URL and model ID with caching"""
    current_time = time.time()
    
    # Only check health if enough time has passed
    if current_time - instance_state["last_health_check"] >= HEALTH_CHECK_INTERVAL:
        is_primary_healthy = await check_instance_health(LLM_URL)
        instance_state["is_primary_healthy"] = is_primary_healthy
        instance_state["last_health_check"] = current_time
        
        if is_primary_healthy:
            instance_state["current_url"] = LLM_URL
            instance_state["current_model"] = MODEL_ID
            return True, (instance_state["current_url"], instance_state["current_model"])
        else:
            instance_state["current_url"] = FALL_BACK_URL
            instance_state["current_model"] = FALL_BACK_MODEL_ID
    
    return False, (instance_state["current_url"], instance_state["current_model"])

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    app.state.client = httpx.AsyncClient(
        timeout=HTTP_TIMEOUT,
        transport=httpx.AsyncHTTPTransport(verify=False),
        http2=True,
        limits=httpx.Limits(max_connections=MAX_CONNECTIONS, max_keepalive_connections=MAX_KEEPALIVE_CONNECTIONS)
    )
    logger.info("Service started successfully with HTTP/2 support and connection pooling")
    
    yield
    
    # Shutdown
    await app.state.client.aclose()
    logger.info("Service shutdown complete")

app = FastAPI(
    title="EternalAI Server",
    description="Server for AI model inference",
    version="2.0.0",
    lifespan=lifespan,
)

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]

# Allow additional origins from environment variable
if os.getenv("ALLOWED_ORIGINS"):
    origins.extend(os.getenv("ALLOWED_ORIGINS").split(","))

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time"],
    max_age=3600,
)

@app.get("/health")
@app.get("/v1/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}

@app.get("/v1/models")
async def models():
    """Models endpoint"""
    return {
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
        }]
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": {"message": "Internal server error", "code": 500}}
    )

@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    chat_request: ChatCompletionRequest
) -> Any:
    """Handle chat completion requests"""
    try:
        is_healthy, (instance_url, model_id) = await get_healthy_instance()

        chat_request.model = model_id
        chat_request.temperature = 0.7
        chat_request.top_k = 20
        chat_request.top_p = 0.8
        chat_request.presence_penalty = 1.5
        chat_request.max_tokens = 8192
        
        if not is_healthy:
            request_payload = {
                "messages": chat_request.messages,
                "model": model_id,
                "max_tokens": 1024,
                "stream": chat_request.stream
            }
        else:
            request_payload = chat_request.dict()

        if request_payload["stream"]:
            async def stream_generator():
                try:
                    async with request.app.state.client.stream(
                        "POST",
                        f"{instance_url}/v1/chat/completions",
                        headers={"Authorization": f"Bearer {API_KEY}"},
                        json=request_payload,
                        timeout=STREAM_TIMEOUT
                    ) as response:
                        if response.status_code != 200:
                            error_text = await response.text()
                            error_msg = {"error": {"message": error_text, "code": response.status_code}}
                            logger.error(f"Streaming error: {response.status_code} - {error_text}")
                            yield f"data: {error_msg}\n\n"
                            yield "data: [DONE]\n\n"
                            return

                        # Process chunks more efficiently
                        async for chunk in response.aiter_bytes():
                            # Decode chunk and split into lines
                            chunk_text = chunk.decode('utf-8')
                            lines = chunk_text.split('\n')
                            
                            # Process each line
                            for line in lines:
                                if line.strip():
                                    yield f"{line}\n\n"
                            
                except httpx.TimeoutException:
                    logger.error("Streaming request timed out")
                    error_msg = {"error": {"message": "Request timed out", "code": 408}}
                    yield f"data: {error_msg}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    logger.error(f"Error in stream: {str(e)}")
                    error_msg = {"error": {"message": str(e), "code": 500}}
                    yield f"data: {error_msg}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            try:
                response = await app.state.client.post(
                    f"{instance_url}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    json=request_payload,
                    timeout=HTTP_TIMEOUT
                )
                if response.status_code != 200:
                    error_text = await response.text()
                    logger.error(f"Backend error: {response.status_code} - {error_text}")
                    if response.status_code == 400:
                        raise HTTPException(status_code=400, detail=error_text)
                    raise HTTPException(status_code=response.status_code, detail=error_text)
                return response.json()
            except httpx.TimeoutException:
                logger.error("Request timed out")
                raise HTTPException(status_code=408, detail="Request timed out")
            except httpx.RequestError as e:
                logger.error(f"Request error: {str(e)}")
                raise HTTPException(status_code=503, detail="Service unavailable")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

if __name__ == "__main__":
    uvicorn.run(
        "apis:app",
        host="0.0.0.0",
        port=8000,
        workers=8
    )