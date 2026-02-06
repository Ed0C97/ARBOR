"""VCR.py fixtures for mocking LLM API calls.

TIER 9 - Point 45: Mocking LLMs (VCR.py/Pytest-Mock)

Records and replays HTTP interactions with LLM APIs for deterministic,
cost-free test runs.

Usage:
    @pytest.mark.vcr()
    async def test_llm_completion():
        result = await gateway.complete(...)
        assert result is not None

First run records cassettes to tests/cassettes/*.yaml
Subsequent runs replay from cassettes.
"""

import os
from pathlib import Path

import pytest

# Try to import VCR, but don't fail if not installed
try:
    import vcr  # noqa: F401
    from vcr import VCR

    HAS_VCR = True
except ImportError:
    HAS_VCR = False
    VCR = None

# Cassette storage directory
CASSETTES_DIR = Path(__file__).parent / "cassettes"


def get_vcr_config():
    """Return VCR configuration for LLM API mocking.

    TIER 9 - Point 45: Test suite runs offline in <10 seconds.
    """
    return {
        "cassette_library_dir": str(CASSETTES_DIR),
        "record_mode": os.getenv("VCR_RECORD_MODE", "none"),  # "new_episodes" to record
        "match_on": ["method", "scheme", "host", "port", "path", "query"],
        "filter_headers": [
            "authorization",
            "x-api-key",
            "api-key",
            "bearer",
        ],
        "filter_post_data_parameters": [
            "api_key",
            "access_token",
        ],
        "decode_compressed_response": True,
        "before_record_request": _filter_sensitive_data,
        "before_record_response": _filter_response_data,
    }


def _filter_sensitive_data(request):
    """Remove sensitive data from recorded requests."""
    # Filter authorization headers
    if "Authorization" in request.headers:
        request.headers["Authorization"] = "Bearer [FILTERED]"
    if "x-api-key" in request.headers:
        request.headers["x-api-key"] = "[FILTERED]"
    return request


def _filter_response_data(response):
    """Filter sensitive data from responses if needed."""
    return response


@pytest.fixture(scope="session")
def vcr_config():
    """Provide VCR configuration to pytest-recording plugin."""
    return get_vcr_config()


@pytest.fixture
def mock_llm_response():
    """Provide a mock LLM completion response."""
    return {
        "id": "chatcmpl-test-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gemini-3-pro-preview",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response from the mocked LLM.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }


@pytest.fixture
def mock_embedding_response():
    """Provide a mock embedding response."""
    return {
        "id": "emb-test-123",
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.1] * 1024,  # Cohere embed-v4.0 dimension
                "index": 0,
            }
        ],
        "model": "embed-v4.0",
        "usage": {
            "prompt_tokens": 5,
            "total_tokens": 5,
        },
    }


@pytest.fixture
def mock_rerank_response():
    """Provide a mock rerank response."""
    return {
        "id": "rerank-test-123",
        "results": [
            {"index": 0, "relevance_score": 0.95},
            {"index": 2, "relevance_score": 0.82},
            {"index": 1, "relevance_score": 0.71},
        ],
        "meta": {
            "api_version": {"version": "2"},
            "billed_units": {"search_units": 1},
        },
    }


# Create cassettes directory if it doesn't exist
CASSETTES_DIR.mkdir(exist_ok=True)
