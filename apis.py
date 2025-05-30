"""
This module provides a FastAPI application that acts as a proxy for chat completion and embedding requests,
forwarding them to an underlying service running on a local port.
"""

import os
import logging
import httpx
import time
import uvicorn
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import sys
from config import CONFIG
from contextlib import asynccontextmanager

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    app.state.client = httpx.AsyncClient(
        timeout=HTTP_TIMEOUT,
        transport=httpx.AsyncHTTPTransport(verify=False),
        http2=True
    )
    logger.info("Service started successfully with HTTP/2 support")
    
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
    return {
        "status": "ok"
    }

@app.get("/v1/models")
async def models():
    """Models endpoint"""
    return {
        "object": "list",
        "data": [{
            "id": CONFIG["model"]["id"],
            "object": "model",
        }]
    }

@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(request: Request, chat_request: ChatCompletionRequest) -> Any:
    """Handle chat completion requests"""
    try:
        chat_request.model = CONFIG["model"]["id"]
        chat_request.temperature = 0.7
        chat_request.top_k = 20
        chat_request.top_p = 0.8
        chat_request.presence_penalty = 1.5
        chat_request.max_tokens = 8192
        instance_url = CONFIG["instance_url"]
        
        request_payload = chat_request.dict()

        if chat_request.stream:
            async def stream_generator():
                try:
                    async with request.app.state.client.stream(
                        "POST",
                        f"{instance_url}/v1/chat/completions",
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

                        buffer = ""
                        async for chunk in response.aiter_bytes():
                            buffer += chunk.decode('utf-8')
                            while '\n' in buffer:
                                line, buffer = buffer.split('\n', 1)
                                if line.strip():
                                    yield f"{line}\n\n"

                        if buffer.strip():
                            yield f"{buffer}\n\n"
                            
                except Exception as e:
                    logger.error(f"Error in stream: {str(e)}")
                    error_msg = {"error": {"message": str(e), "code": 500}}
                    yield f"data: {error_msg}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream"
            )
        else:
            response = await app.state.client.post(
                f"{instance_url}/v1/chat/completions",
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
        port=CONFIG.get("proxy_port", 65534),
        workers=CONFIG.get("workers", 1)
    )