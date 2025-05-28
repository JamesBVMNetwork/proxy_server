# OpenAI API Proxy Server

A simple proxy server that forwards requests to OpenAI's API endpoints.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python apis.py [--port PORT] [--host HOST]
```

Command line options:
- `--port PORT`: Port to run the server on (default: 8000)
- `--host HOST`: Host to run the server on (default: 0.0.0.0)

Examples:
```bash
# Run on default port 8000
python apis.py

# Run on custom port 9000
python apis.py --port 9000

# Run on localhost only
python apis.py --host 127.0.0.1
```

The server will start on the specified host and port (default: http://localhost:8000)

## Usage

The proxy server exposes the following endpoint:

### POST /v1/chat/completions

Make requests to the proxy server exactly as you would to the OpenAI API. Just replace the base URL with your proxy server URL.

Example:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_OPENAI_API_KEY" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Features

- Supports all ChatCompletion parameters
- Handles streaming responses
- Forwards OpenAI API errors
- Automatic request validation
- Interactive API documentation at `/docs`

## API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.