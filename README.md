# Raven Chat API

A FastAPI service that processes natural language queries to locate and extract tables/images from a manual, returning structured responses with citations.

## Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository and navigate to the project directory:
```bash
cd ravenappdev-assignment
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Server

Start the FastAPI server with auto-reload:
```bash
fastapi dev src/main.py
```

The server will start at `http://localhost:8000`

### API Documentation

Once the server is running, you can access:
- Interactive API docs: `http://localhost:8000/docs`
- Alternative API docs: `http://localhost:8000/redoc`

## API Usage

### Chat Endpoint

**POST** `/chat`

Request body:
```json
{
  "query": "Show me the sizing factors for liquids"
}
```

Response:
```json
{
  "content": "## Liquid Valve Sizing Factors\n\nChapter 3 covers liquid valve sizing...",
  "citations": [
    {
      "page_no": 23,
      "bbox": {
        "top_left_x": 100,
        "top_left_y": 200,
        "width": 800,
        "height": 600
      },
      "confidence": 0.88
    }
  ],
  "status": "success"
}
```

### Example Queries

- "Can you pull up the comparison of actuator types?"
- "Show me the sizing factors for liquids."
- "I need the noise level reference values."
- "Do we have a figure that explains cavitation?"
- "Show me the overall oil & gas value chain diagram."

## Testing

### Unit Tests

Unit tests are designed to test individual components in isolation using mocks and test fixtures. They don't require the server to be running.

Run all unit tests:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=src tests/
```

Run tests with verbose output:
```bash
pytest -v
```

### Probers (Integration Tests)

Probers are integration tests that verify the API is working correctly by making actual HTTP requests to a running server. They test the system externally as a black box.

1. Start the server in one terminal:
```bash
fastapi dev src/main.py
```

2. In another terminal, run the probers:
```bash
cd probers
pytest
```

Or run probers from the root directory:
```bash
pytest probers/
```

**Key Differences:**
- **Unit Tests**: Test internal logic with mocks, no server needed, fast execution
- **Probers**: Test the API externally via HTTP, require running server, verify end-to-end functionality

## Docker Support

### Building the Docker Image

```bash
docker build -t raven-chat-api .
```

### Running the Docker Container

```bash
docker run -p 8000:8000 raven-chat-api
```

The API will be accessible at `http://localhost:8000`

## Project Structure

```
ravenappdev-assignment/
├── src/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   └── models.py            # Pydantic models
├── tests/
│   ├── __init__.py
│   └── test_api.py          # Unit tests
├── probers/
│   ├── test_chat_prober.py  # Integration tests
│   └── pytest.ini           # Prober-specific pytest config
├── requirements.txt         # Python dependencies
├── pytest.ini              # Root pytest config (excludes probers)
├── Dockerfile              # Docker configuration
├── CLAUDE.md               # Claude Code guidance
└── README.md               # This file
```

## Development Notes

This is a placeholder implementation with mock responses. The actual implementation will include:
- Document indexing using FAISS
- Semantic search and retrieval
- Mapping to actual page numbers and bounding boxes from `mmd_lines_data.json`
- Processing of the `manual.mmd` content

## TODO

- [ ] Test auto-reload in docker image, I am using FastAPI directly.