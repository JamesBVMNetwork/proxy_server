"""
This module provides a FastAPI application that acts as a proxy for chat completion and embedding requests,
forwarding them to an underlying service running on a local port.
"""

import os
import logging
import httpx
import time
import uvicorn
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import sys
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from dataclasses import dataclass
from enum import Enum

load_dotenv()

class ProviderType(Enum):
    SELF_HOSTED = "self_hosted"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    DEEPINFRA = "deepinfra"

@dataclass
class ProviderConfig:
    """Configuration for an LLM provider"""
    name: str
    type: ProviderType
    base_url: str
    model_id: str
    api_key: Optional[str] = None
    priority: int = 0
    timeout: float = 60.0
    max_tokens: int = 8192
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    presence_penalty: float = 1.5
    frequency_penalty: float = 0.0
    min_p: float = 0.0
    seed: int = 0

def load_provider_configs() -> List[ProviderConfig]:
    """Load provider configurations from environment variables"""
    providers = []
    
    # Self-hosted provider (highest priority)
    if os.getenv("SELF_HOSTED_URL") and os.getenv("SELF_HOSTED_MODEL"):
        providers.append(ProviderConfig(
            name="Self-hosted",
            type=ProviderType.SELF_HOSTED,
            base_url=os.getenv("SELF_HOSTED_URL"),
            model_id=os.getenv("SELF_HOSTED_MODEL"),
            priority=0
        ))
    
    # OpenAI provider
    if os.getenv("OPENAI_API_KEY"):
        providers.append(ProviderConfig(
            name="OpenAI",
            type=ProviderType.OPENAI,
            base_url="https://api.openai.com/v1",
            model_id=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            api_key=os.getenv("OPENAI_API_KEY"),
            priority=1
        ))
    
    # OpenRouter provider
    if os.getenv("OPENROUTER_API_KEY"):
        providers.append(ProviderConfig(
            name="OpenRouter",
            type=ProviderType.OPENROUTER,
            base_url="https://openrouter.ai/api/v1",
            model_id=os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo"),
            api_key=os.getenv("OPENROUTER_API_KEY"),
            priority=2
        ))
    
    # DeepInfra provider
    if os.getenv("DEEPINFRA_API_KEY"):
        providers.append(ProviderConfig(
            name="DeepInfra",
            type=ProviderType.DEEPINFRA,
            base_url="https://api.deepinfra.com/v1",
            model_id=os.getenv("DEEPINFRA_MODEL", "meta-llama/Llama-2-70b-chat-hf"),
            api_key=os.getenv("DEEPINFRA_API_KEY"),
            priority=3
        ))
    
    return sorted(providers, key=lambda x: x.priority)

# Load provider configurations
PROVIDERS = load_provider_configs()

# Constants
HTTP_TIMEOUT = 60.0
STREAM_TIMEOUT = 600.0
HEALTH_CHECK_CACHE_SECONDS = 10.0
MAX_CONNECTIONS = 100
MAX_KEEPALIVE_CONNECTIONS = 20
KEEPALIVE_TIMEOUT = 5.0

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

# Set up logging (simplified)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING if os.getenv("ENV") == "production" else logging.INFO)
logger.handlers.clear()
handler = ErrorHandlingStreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Configure uvicorn access logger
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.handlers.clear()
uvicorn_access_handler = ErrorHandlingStreamHandler(sys.stderr)
uvicorn_access_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
uvicorn_access_logger.addHandler(uvicorn_access_handler)

class InstanceState:
    """Encapsulates instance health and model state with health check caching."""
    def __init__(self):
        self.provider_states = {}
        for provider in PROVIDERS:
            self.provider_states[provider.name] = {
                "last_health_check": 0.0,
                "is_healthy": True,
                "_health_cache": None,
                "_health_cache_time": 0.0
            }

    async def check_instance_health(self, provider: ProviderConfig) -> bool:
        """Check health of a specific provider with caching."""
        state = self.provider_states[provider.name]
        now = time.time()
        
        # Return cached health status if available and not expired
        if state["_health_cache"] is not None and (now - state["_health_cache_time"]) < HEALTH_CHECK_CACHE_SECONDS:
            return state["_health_cache"]
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Different health check endpoints for different providers
                if provider.type == ProviderType.SELF_HOSTED:
                    health_url = f"{provider.base_url}/health"
                elif provider.type == ProviderType.OPENAI:
                    health_url = f"{provider.base_url}/models"
                elif provider.type == ProviderType.OPENROUTER:
                    health_url = f"{provider.base_url}/models"
                elif provider.type == ProviderType.DEEPINFRA:
                    health_url = f"{provider.base_url}/models"
                else:
                    health_url = f"{provider.base_url}/health"
                
                headers = {}
                if provider.api_key:
                    headers["Authorization"] = f"Bearer {provider.api_key}"
                
                response = await client.get(health_url, headers=headers)
                healthy = response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed for {provider.name}: {str(e)}")
            healthy = False
        
        # Update cache
        state["_health_cache"] = healthy
        state["_health_cache_time"] = now
        state["is_healthy"] = healthy
        return healthy

    async def get_healthy_provider(self) -> Optional[ProviderConfig]:
        """Get the first healthy provider based on priority."""
        for provider in PROVIDERS:
            if await self.check_instance_health(provider):
                return provider
        return None

    def get_provider_state(self, provider_name: str) -> Dict[str, Any]:
        """Get the current state of a provider."""
        return self.provider_states.get(provider_name, {})

instance_state = InstanceState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    app.state.client = httpx.AsyncClient(
        timeout=HTTP_TIMEOUT,
        transport=httpx.AsyncHTTPTransport(verify=False),
        http2=True,
        limits=httpx.Limits(max_connections=MAX_CONNECTIONS, max_keepalive_connections=MAX_KEEPALIVE_CONNECTIONS)
    )
    logger.info("Service started successfully with HTTP/2 support and connection pooling")
    yield
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
if os.getenv("ALLOWED_ORIGINS"):
    origins.extend([o.strip() for o in os.getenv("ALLOWED_ORIGINS").split(",") if o.strip()])

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
async def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}

@app.get("/v1/models")
async def models() -> Dict[str, Any]:
    """Models endpoint."""
    return {
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
        }]
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": {"message": "Internal server error", "code": 500}}
    )

def build_request_payload(chat_request: ChatCompletionRequest, model_id: str, is_healthy: bool) -> Dict[str, Any]:
    """Builds the payload for the backend request without mutating the input object."""
    logger.info("Calling build_request_payload")
    if not is_healthy:
        return {
            "messages": chat_request.messages,
            "model": model_id,
            "max_tokens": 1024,
            "stream": chat_request.stream,
            "seed": 0,
            "tools": chat_request.tools,
            "tool_choice": chat_request.tool_choice,
        }
    # Build payload dict with defaults
    payload = chat_request.dict()
    payload["model"] = model_id
    payload["temperature"] = 0.7
    payload["top_k"] = 20
    payload["top_p"] = 0.8
    payload["presence_penalty"] = 1.5
    payload["max_tokens"] = 8192
    payload["min_p"] = 0.0
    payload["seed"] = 0
    payload["frequency_penalty"] = 0.0
    # Remove tools/tool_choice if not present
    if not payload.get("tools"):
        payload.pop("tools", None)
        payload.pop("tool_choice", None)
    elif not payload.get("tool_choice"):
        payload.pop("tool_choice", None)
    return payload

@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    chat_request: ChatCompletionRequest
) -> Any:
    """Handle chat completion requests."""
    async def handle_streaming(provider: ProviderConfig, request_payload: Dict[str, Any]) -> StreamingResponse:
        async def stream_generator():
            try:
                headers = {"Content-Type": "application/json"}
                if provider.api_key:
                    headers["Authorization"] = f"Bearer {provider.api_key}"
                
                async with request.app.state.client.stream(
                    "POST",
                    f"{provider.base_url}/v1/chat/completions",
                    headers=headers,
                    json=request_payload,
                    timeout=provider.timeout
                ) as response:
                    if response.status_code != 200:
                        error_bytes = await response.aread()
                        error_text = error_bytes.decode('utf-8', errors='replace')
                        error_msg = f"data: {{\"error\":{{\"message\":\"{error_text}\",\"code\":{response.status_code}}}}}\n\n"
                        logger.error(f"Streaming error from {provider.name}: {response.status_code} - {error_text}")
                        yield error_msg
                        return

                    buffer = ""
                    async for chunk in response.aiter_bytes():
                        buffer += chunk.decode('utf-8')
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            if line.strip():
                                try:
                                    json_str = line.strip()
                                    if json_str.startswith('data: '):
                                        json_str = json_str[6:]
                                    if json_str.strip() == '[DONE]':
                                        yield 'data: [DONE]\n\n'
                                        continue
                                    chunk_obj = ChatCompletionChunk.parse_raw(json_str)
                                    yield f"data: {chunk_obj.json()}\n\n"
                                except Exception as e:
                                    logger.error(f"Failed to parse streaming chunk from {provider.name}: {e}; line: {line}")
                                    yield f"{line}\n\n"
                    if buffer.strip():
                        try:
                            json_str = buffer.strip()
                            if json_str.startswith('data: '):
                                json_str = json_str[6:]
                            if json_str.strip() != '[DONE]':
                                chunk_obj = ChatCompletionChunk.parse_raw(json_str)
                                yield f"data: {chunk_obj.json()}\n\n"
                            else:
                                yield 'data: [DONE]\n\n'
                        except Exception as e:
                            logger.error(f"Failed to parse trailing streaming chunk from {provider.name}: {e}; buffer: {buffer}")
                            yield f"{buffer}\n\n"
            except httpx.TimeoutException:
                logger.error(f"Streaming request timed out for {provider.name}")
                yield f"data: {{\"error\":{{\"message\":\"Request timed out\",\"code\":408}}}}\n\n"
            except Exception as e:
                logger.error(f"Error during streaming from {provider.name}: {e}")
                yield f"data: {{\"error\":{{\"message\":\"{str(e)}\",\"code\":500}}}}\n\n"
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

    async def handle_non_streaming(provider: ProviderConfig, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            headers = {"Content-Type": "application/json"}
            if provider.api_key:
                headers["Authorization"] = f"Bearer {provider.api_key}"
            
            response = await app.state.client.post(
                f"{provider.base_url}/v1/chat/completions",
                headers=headers,
                json=request_payload,
                timeout=provider.timeout
            )
            if response.status_code != 200:
                error_text = await response.text()
                logger.error(f"Backend error from {provider.name}: {response.status_code} - {error_text}")
                if response.status_code == 400:
                    raise HTTPException(status_code=400, detail=error_text)
                raise HTTPException(status_code=response.status_code, detail=error_text)
            try:
                completion_obj = ChatCompletionResponse.parse_obj(response.json())
                return completion_obj.dict()
            except Exception as e:
                logger.error(f"Failed to parse ChatCompletionResponse from {provider.name}: {e}")
                raise HTTPException(status_code=500, detail="Invalid response format from backend")
        except httpx.TimeoutException:
            logger.error(f"Request timed out for {provider.name}")
            raise HTTPException(status_code=408, detail="Request timed out")
        except httpx.RequestError as e:
            logger.error(f"Request error from {provider.name}: {str(e)}")
            raise HTTPException(status_code=503, detail="Service unavailable")

    try:
        # Get the first healthy provider
        provider = await instance_state.get_healthy_provider()
        if not provider:
            raise HTTPException(status_code=503, detail="No healthy providers available")
        
        # Build request payload with provider-specific settings
        request_payload = build_request_payload(chat_request, provider.model_id, True)
        request_payload.update({
            "temperature": provider.temperature,
            "top_p": provider.top_p,
            "top_k": provider.top_k,
            "presence_penalty": provider.presence_penalty,
            "max_tokens": provider.max_tokens,
            "min_p": provider.min_p,
            "seed": provider.seed,
            "frequency_penalty": provider.frequency_penalty
        })
        
        if request_payload.get("stream"):
            return await handle_streaming(provider, request_payload)
        else:
            return await handle_non_streaming(provider, request_payload)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

if __name__ == "__main__":
    uvicorn.run(
        "apis:app",
        host="0.0.0.0",
        port=6060,
        workers=8
    )