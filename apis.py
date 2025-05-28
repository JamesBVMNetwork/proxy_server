import os
import logging
import argparse
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
import httpx
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OpenAI API Proxy")

# OpenAI API base URL
OPENAI_API_BASE = "https://api.openai.com"
STREAM_TIMEOUT = 60.0  # seconds

# Initialize the httpx client at startup
@app.on_event("startup")
async def startup_event():
    app.state.client = httpx.AsyncClient()

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.client.aclose()

class ChatCompletionRequest(BaseModel):
    model: str
    messages: Any
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = None
    stop: Optional[list[str]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

async def stream_generator(response: httpx.Response):
    """
    Generator for streaming responses from OpenAI.
    Yields chunks of data as they are received, formatted for SSE.
    """
    try:
        if response.status_code != 200:
            error_text = await response.text()
            error_msg = f"data: {{\"error\":{{\"message\":\"{error_text}\",\"code\":{response.status_code}}}}}\n\n"
            logger.error(f"Streaming error: {response.status_code} - {error_text}")
            yield error_msg
            return

        buffer = ""
        async for chunk in response.aiter_bytes():
            buffer += chunk.decode('utf-8')
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                if line.strip():
                    yield f"{line}\n\n"
        
        # Process any remaining data in the buffer
        if buffer.strip():
            yield f"{buffer}\n\n"
            
    except Exception as e:
        logger.error(f"Error during streaming: {e}")
        yield f"data: {{\"error\":{{\"message\":\"{str(e)}\",\"code\":500}}}}\n\n"

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None)
):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header is required")

    headers = {
        "Authorization": authorization,
        "Content-Type": "application/json"
    }

    url = f"{OPENAI_API_BASE}/v1/chat/completions"
    
    try:
        if request.stream:
            response = await app.state.client.stream(
                "POST",
                url,
                headers=headers,
                json=request.model_dump(exclude_none=True),
                timeout=STREAM_TIMEOUT
            )
            return StreamingResponse(
                stream_generator(response),
                media_type="text/event-stream"
            )
        else:
            response = await app.state.client.post(
                url,
                headers=headers,
                json=request.model_dump(exclude_none=True),
                timeout=STREAM_TIMEOUT
            )
            return response.json()
            
    except Exception as e:
        logger.error(f"Request error: {e}")
        raise HTTPException(status_code=502, detail=f"Error communicating with OpenAI: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='OpenAI API Proxy Server')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on (default: 8000)')
    parser.add_argument('--host', type=str, default="0.0.0.0", help='Host to run the server on (default: 0.0.0.0)')
    args = parser.parse_args()
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)