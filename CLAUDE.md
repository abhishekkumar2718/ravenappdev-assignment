# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Background

I am working on this project as a take-home assignment. This means you should prioritize **first principles thinking**, **velocity**, **simplicity** over **productionizing code**.

The deadline is 24 hours, and I would be able to give 4-6 hours to the assignment at most. Prioritize solutions that fit within this time frame.

## Problem

You are given a manual on control valves and are building a chatbot to answer queries that technicians might have.

Key files:
- `manual.mmd`: contains the transcription of the manual
- `mmd_lines_data.json`: contains the page number and bounding box co-ordinates associated with text/images and tables.

In particular, you want the chatbot to prioritize:
- **Accuracy**: Incorrect answers can potentially be life-harming. It is better to respond that you don't know rather than give incorrect answers.
- **Conciseness**: Technicians have limited time and prefer concise answers over elaborate answers
- **Transparency**: The responses must be transparent and contain references to the user manual, as well as explain their reasoning steps to build trust with technicians.

## Task Requirements

The backend service must:
1. **Locate** relevant tables/images for natural language operator queries
2. **Return** the table/image content
3. **Include citation data**: page number and bounding box coordinates
4. Handle ambiguity by returning top candidates
5. Return `"insufficient_info"` if no confident match

## Example Queries
- "Can you pull up the comparison of actuator types?"
- "Show me the sizing factors for liquids."
- "I need the noise level reference values."
- "Do we have a figure that explains cavitation?"
- "Show me the overall oil & gas value chain diagram."

## Code Quality Requirements

The assignment emphasizes:
- **Modular, well-structured code** with clean separation of concerns
- Separate modules for indexing, retrieval, and response formatting
- Meaningful function/module boundaries
- Readable, maintainable code (not just a working script)

## Available Commands

### Running the Server
```bash
fastapi dev src/main.py
```

### Running Tests

#### Unit Tests
Unit tests are designed to test individual components in isolation using mocks and test fixtures. They don't require external services to be running.

```bash
# Run all unit tests (excludes probers)
pytest

# Run tests with coverage
pytest --cov=src tests/

# Run tests with verbose output
pytest -v
```

#### Probers (Integration Tests)
Probers are integration tests that verify the API is working correctly by making actual HTTP requests to a running server. They test the system externally as a black box.

```bash
# First, start the server in one terminal
fastapi dev src/main.py

# Then run probers in another terminal
cd probers
pytest

# Or from the root directory
pytest probers/
```

**Key Differences:**
- **Unit Tests**: Test internal logic with mocks, no server needed, fast execution
- **Probers**: Test the API externally via HTTP, require running server, verify end-to-end functionality

### Docker Commands
```bash
# Build Docker image
docker build -t raven-chat-api .

# Run Docker container
docker run -p 8000:8000 raven-chat-api
```