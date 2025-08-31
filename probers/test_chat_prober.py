import pytest
import requests
import time
import sys


# Configuration
API_BASE_URL = "http://localhost:8000"
TIMEOUT = 10  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

@pytest.fixture(scope="module")
def api_url():
    """Provide the API URL and ensure server is running"""
    return API_BASE_URL


class TestChatEndpointProber:
    """Prober tests for the /chat endpoint"""
    
    def test_chat_endpoint_available(self, api_url):
        """Test that the /chat endpoint is available and responds"""
        response = requests.post(
            f"{api_url}/chat",
            json={"query": "What are eccentric plug valves"},
            timeout=TIMEOUT
        )

        assert response.status_code in [200, 422], f"Unexpected status code: {response.status_code}"

        print(response.json())    
