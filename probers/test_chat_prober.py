import pytest
import requests
import time
import sys


# Configuration
API_BASE_URL = "http://localhost:8000"
TIMEOUT = 10  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds


def wait_for_server(url, timeout=30):
    """Wait for the server to be available"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(1)
    return False


@pytest.fixture(scope="module")
def api_url():
    """Provide the API URL and ensure server is running"""
    if not wait_for_server(API_BASE_URL):
        pytest.skip(f"API server not available at {API_BASE_URL}")
    return API_BASE_URL


class TestChatEndpointProber:
    """Prober tests for the /chat endpoint"""
    
    def test_chat_endpoint_available(self, api_url):
        """Test that the /chat endpoint is available and responds"""
        response = requests.post(
            f"{api_url}/chat",
            json={"query": "test query"},
            timeout=TIMEOUT
        )
        assert response.status_code in [200, 422], f"Unexpected status code: {response.status_code}"    
