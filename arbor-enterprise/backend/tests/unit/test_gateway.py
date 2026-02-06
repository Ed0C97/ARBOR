"""Unit tests for the LLM Gateway module.

Tests the LLMGateway class including provider selection, completion,
embedding, reranking, batch operations, and fallback behavior with
all external calls (LiteLLM, Cohere) mocked out.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ═══════════════════════════════════════════════════════════════════════════
# Helper: build a gateway with mocked config
# ═══════════════════════════════════════════════════════════════════════════


def _make_gateway(
    google_key="test-google-key",
    cohere_key="test-cohere-key",
    openai_key="",
    anthropic_key="",
    groq_key="",
):
    """Construct an LLMGateway with controlled settings."""
    with patch("app.llm.gateway.settings") as mock_settings, \
         patch("app.llm.gateway.Router") as mock_router_cls, \
         patch("app.llm.gateway.get_cohere_client") as mock_get_cohere:

        mock_settings.google_api_key = google_key
        mock_settings.google_model = "gemini-3-pro-preview"
        mock_settings.cohere_api_key = cohere_key
        mock_settings.cohere_embedding_model = "embed-v4.0"
        mock_settings.cohere_rerank_model = "rerank-v4.0-pro"
        mock_settings.openai_api_key = openai_key
        mock_settings.openai_model = "gpt-4o"
        mock_settings.anthropic_api_key = anthropic_key
        mock_settings.anthropic_model = "claude-sonnet-4-20250514"
        mock_settings.groq_api_key = groq_key
        mock_settings.groq_model = "llama-3.3-70b-versatile"
        mock_settings.mistral_api_key = ""

        mock_router_cls.return_value = MagicMock()
        mock_get_cohere.return_value = MagicMock()

        from app.llm.gateway import LLMGateway
        gw = LLMGateway()
        return gw


# ═══════════════════════════════════════════════════════════════════════════
# Model Selection Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestModelSelection:
    """Tests for LLMGateway._select_model()."""

    def setup_method(self):
        self.gw = _make_gateway()

    def test_complex_maps_to_primary(self):
        assert self.gw._select_model("complex") == "primary"

    def test_synthesis_maps_to_primary(self):
        assert self.gw._select_model("synthesis") == "primary"

    def test_simple_maps_to_simple(self):
        assert self.gw._select_model("simple") == "simple"

    def test_extraction_maps_to_simple(self):
        assert self.gw._select_model("extraction") == "simple"

    def test_classification_maps_to_fast(self):
        assert self.gw._select_model("classification") == "fast"

    def test_fast_maps_to_fast(self):
        assert self.gw._select_model("fast") == "fast"

    def test_vision_maps_to_vision(self):
        assert self.gw._select_model("vision") == "vision"

    def test_unknown_task_defaults_to_primary(self):
        assert self.gw._select_model("nonexistent_task") == "primary"


# ═══════════════════════════════════════════════════════════════════════════
# Completion Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestComplete:
    """Tests for LLMGateway.complete()."""

    @pytest.mark.asyncio
    async def test_complete_uses_router_when_available(self):
        """complete() delegates to router.acompletion when router is set."""
        gw = _make_gateway()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello world"))]
        gw.router = AsyncMock()
        gw.router.acompletion.return_value = mock_response

        result = await gw.complete(
            messages=[{"role": "user", "content": "Hi"}],
            task_type="simple",
        )

        assert result == "Hello world"
        gw.router.acompletion.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("app.llm.gateway.acompletion", new_callable=AsyncMock)
    async def test_complete_falls_back_to_acompletion(self, mock_acompletion):
        """complete() calls litellm.acompletion when router is None."""
        gw = _make_gateway()
        gw.router = None

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Fallback response"))]
        mock_acompletion.return_value = mock_response

        result = await gw.complete(
            messages=[{"role": "user", "content": "Hi"}],
            model="primary",
        )

        assert result == "Fallback response"
        mock_acompletion.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_complete_passes_temperature(self):
        """complete() forwards the temperature parameter."""
        gw = _make_gateway()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        gw.router = AsyncMock()
        gw.router.acompletion.return_value = mock_response

        await gw.complete(
            messages=[{"role": "user", "content": "test"}],
            temperature=0.2,
        )

        call_kwargs = gw.router.acompletion.call_args.kwargs
        assert call_kwargs["temperature"] == 0.2

    @pytest.mark.asyncio
    async def test_complete_passes_max_tokens(self):
        """complete() forwards max_tokens when provided."""
        gw = _make_gateway()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        gw.router = AsyncMock()
        gw.router.acompletion.return_value = mock_response

        await gw.complete(
            messages=[{"role": "user", "content": "test"}],
            max_tokens=500,
        )

        call_kwargs = gw.router.acompletion.call_args.kwargs
        assert call_kwargs["max_tokens"] == 500

    @pytest.mark.asyncio
    async def test_complete_passes_response_format(self):
        """complete() forwards response_format for JSON mode."""
        gw = _make_gateway()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"key": 1}'))]
        gw.router = AsyncMock()
        gw.router.acompletion.return_value = mock_response

        await gw.complete(
            messages=[{"role": "user", "content": "give json"}],
            response_format={"type": "json_object"},
        )

        call_kwargs = gw.router.acompletion.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_complete_raises_on_provider_error(self):
        """complete() re-raises when the LLM call fails."""
        gw = _make_gateway()
        gw.router = AsyncMock()
        gw.router.acompletion.side_effect = Exception("Rate limit exceeded")

        with pytest.raises(Exception, match="Rate limit exceeded"):
            await gw.complete(messages=[{"role": "user", "content": "Hi"}])


# ═══════════════════════════════════════════════════════════════════════════
# complete_json / complete_vision convenience methods
# ═══════════════════════════════════════════════════════════════════════════


class TestConvenienceMethods:
    """Tests for complete_json() and complete_vision()."""

    @pytest.mark.asyncio
    async def test_complete_json_uses_low_temperature(self):
        """complete_json() calls complete with temperature=0.3."""
        gw = _make_gateway()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{}'))]
        gw.router = AsyncMock()
        gw.router.acompletion.return_value = mock_response

        await gw.complete_json(messages=[{"role": "user", "content": "extract"}])

        call_kwargs = gw.router.acompletion.call_args.kwargs
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_complete_vision_uses_vision_model(self):
        """complete_vision() selects the vision model."""
        gw = _make_gateway()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="I see a cat"))]
        gw.router = AsyncMock()
        gw.router.acompletion.return_value = mock_response

        result = await gw.complete_vision(
            messages=[{"role": "user", "content": "describe this image"}],
        )

        assert result == "I see a cat"
        call_kwargs = gw.router.acompletion.call_args.kwargs
        assert call_kwargs["model"] == "vision"


# ═══════════════════════════════════════════════════════════════════════════
# Embedding Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestEmbedding:
    """Tests for LLMGateway.get_embedding() and get_query_embedding()."""

    @pytest.mark.asyncio
    async def test_get_embedding_with_cohere(self):
        """get_embedding() calls async Cohere client when available."""
        gw = _make_gateway()

        mock_cohere = AsyncMock()
        mock_embeddings = MagicMock()
        mock_embeddings.float_ = [[0.1, 0.2, 0.3]]
        mock_response = MagicMock()
        mock_response.embeddings = mock_embeddings
        mock_cohere.embed.return_value = mock_response
        gw._async_cohere = mock_cohere
        gw._async_cohere_initialized = True

        result = await gw.get_embedding("test text")

        assert result == [0.1, 0.2, 0.3]
        mock_cohere.embed.assert_awaited_once()
        call_kwargs = mock_cohere.embed.call_args.kwargs
        assert call_kwargs["input_type"] == "search_document"

    @pytest.mark.asyncio
    async def test_get_query_embedding_uses_search_query_type(self):
        """get_query_embedding() uses input_type='search_query'."""
        gw = _make_gateway()

        mock_cohere = AsyncMock()
        mock_embeddings = MagicMock()
        mock_embeddings.float_ = [[0.4, 0.5, 0.6]]
        mock_response = MagicMock()
        mock_response.embeddings = mock_embeddings
        mock_cohere.embed.return_value = mock_response
        gw._async_cohere = mock_cohere
        gw._async_cohere_initialized = True

        result = await gw.get_query_embedding("search for restaurants")

        assert result == [0.4, 0.5, 0.6]
        call_kwargs = mock_cohere.embed.call_args.kwargs
        assert call_kwargs["input_type"] == "search_query"

    @pytest.mark.asyncio
    @patch("app.llm.gateway.aembedding", new_callable=AsyncMock)
    async def test_get_embedding_fallback_to_openai(self, mock_aembedding):
        """get_embedding() falls back to OpenAI when Cohere is not available."""
        gw = _make_gateway(cohere_key="")
        gw._async_cohere = None
        gw._async_cohere_initialized = True

        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.7, 0.8]}]
        mock_aembedding.return_value = mock_response

        result = await gw.get_embedding("fallback text")

        assert result == [0.7, 0.8]
        mock_aembedding.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_embedding_raises_when_no_provider(self):
        """get_embedding() raises RuntimeError when no provider is available."""
        gw = _make_gateway(cohere_key="")
        gw._async_cohere = None
        gw._async_cohere_initialized = True

        with patch("app.llm.gateway.aembedding", new_callable=AsyncMock) as mock_ae:
            mock_ae.side_effect = Exception("OpenAI down")

            with pytest.raises(RuntimeError, match="No embedding provider available"):
                await gw.get_embedding("doomed text")


# ═══════════════════════════════════════════════════════════════════════════
# Batch Embedding Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBatchEmbedding:
    """Tests for LLMGateway.get_embeddings_batch()."""

    @pytest.mark.asyncio
    async def test_batch_embedding_returns_empty_for_empty_input(self):
        """get_embeddings_batch([]) returns an empty list."""
        gw = _make_gateway()
        result = await gw.get_embeddings_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_batch_embedding_chunks_large_batches(self):
        """get_embeddings_batch() splits texts into chunks of 96."""
        gw = _make_gateway()

        mock_cohere = AsyncMock()
        # Each batch returns a list of embeddings matching its size
        async def fake_embed(**kwargs):
            texts = kwargs.get("texts", [])
            resp = MagicMock()
            resp.embeddings = MagicMock()
            resp.embeddings.float_ = [[1.0] * 3 for _ in texts]
            return resp

        mock_cohere.embed = fake_embed
        gw._async_cohere = mock_cohere
        gw._async_cohere_initialized = True

        # Send 200 texts — should split into 3 chunks (96, 96, 8)
        texts = [f"text_{i}" for i in range(200)]
        result = await gw.get_embeddings_batch(texts)

        assert len(result) == 200
        assert all(len(emb) == 3 for emb in result)


# ═══════════════════════════════════════════════════════════════════════════
# Reranking Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestReranking:
    """Tests for LLMGateway.rerank() and rerank_results()."""

    @pytest.mark.asyncio
    async def test_rerank_with_cohere(self):
        """rerank() calls Cohere rerank and returns structured results."""
        gw = _make_gateway()

        mock_cohere = AsyncMock()
        r1 = MagicMock(index=1, relevance_score=0.95)
        r2 = MagicMock(index=0, relevance_score=0.80)
        mock_response = MagicMock()
        mock_response.results = [r1, r2]
        mock_cohere.rerank.return_value = mock_response
        gw._async_cohere = mock_cohere
        gw._async_cohere_initialized = True

        documents = ["Doc A", "Doc B"]
        result = await gw.rerank("query", documents, top_n=2)

        assert len(result) == 2
        assert result[0]["relevance_score"] == 0.95
        assert result[0]["document"] == "Doc B"  # index 1 -> Doc B

    @pytest.mark.asyncio
    async def test_rerank_fallback_when_cohere_unavailable(self):
        """rerank() returns original order when Cohere is not configured."""
        gw = _make_gateway(cohere_key="")
        gw._async_cohere = None
        gw._async_cohere_initialized = True

        documents = ["Doc A", "Doc B", "Doc C"]
        result = await gw.rerank("query", documents, top_n=2)

        assert len(result) == 2
        assert result[0]["index"] == 0
        assert result[0]["relevance_score"] == 1.0

    @pytest.mark.asyncio
    async def test_rerank_fallback_on_api_error(self):
        """rerank() gracefully degrades when Cohere API raises."""
        gw = _make_gateway()

        mock_cohere = AsyncMock()
        mock_cohere.rerank.side_effect = Exception("API error")
        gw._async_cohere = mock_cohere
        gw._async_cohere_initialized = True

        documents = ["Doc A", "Doc B"]
        result = await gw.rerank("query", documents, top_n=2)

        # Should return original order as fallback
        assert len(result) == 2
        assert result[0]["relevance_score"] == 1.0

    @pytest.mark.asyncio
    async def test_rerank_results_preserves_original_objects(self):
        """rerank_results() adds relevance_score and original_rank to result dicts."""
        gw = _make_gateway()

        mock_cohere = AsyncMock()
        r1 = MagicMock(index=1, relevance_score=0.99)
        r2 = MagicMock(index=0, relevance_score=0.75)
        mock_response = MagicMock()
        mock_response.results = [r1, r2]
        mock_cohere.rerank.return_value = mock_response
        gw._async_cohere = mock_cohere
        gw._async_cohere_initialized = True

        results = [
            {"name": "Alpha", "category": "food", "description": "A restaurant"},
            {"name": "Beta", "category": "fashion", "description": "A boutique"},
        ]

        reranked = await gw.rerank_results("query", results, top_n=2)

        assert len(reranked) == 2
        # First result should be index=1 (Beta) with score 0.99
        assert reranked[0]["name"] == "Beta"
        assert reranked[0]["relevance_score"] == 0.99
        assert reranked[0]["original_rank"] == 2

    @pytest.mark.asyncio
    async def test_rerank_results_empty_input(self):
        """rerank_results([]) returns empty list without calling Cohere."""
        gw = _make_gateway()
        result = await gw.rerank_results("query", [])
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════
# Chunker Utility Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestChunker:
    """Tests for the chunker() batch utility function."""

    def test_chunker_splits_evenly(self):
        from app.llm.gateway import chunker
        result = chunker([1, 2, 3, 4, 5, 6], 3)
        assert result == [[1, 2, 3], [4, 5, 6]]

    def test_chunker_handles_remainder(self):
        from app.llm.gateway import chunker
        result = chunker([1, 2, 3, 4, 5], 2)
        assert result == [[1, 2], [3, 4], [5]]

    def test_chunker_single_chunk(self):
        from app.llm.gateway import chunker
        result = chunker([1, 2, 3], 10)
        assert result == [[1, 2, 3]]

    def test_chunker_empty_input(self):
        from app.llm.gateway import chunker
        result = chunker([], 5)
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════
# Gateway Singleton & Lifecycle Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestGatewaySingleton:
    """Tests for get_llm_gateway() singleton and close()."""

    def test_gateway_has_router_when_keys_configured(self):
        """LLMGateway initializes with a router when API keys are present."""
        gw = _make_gateway(google_key="test-key")
        assert gw.router is not None

    def test_gateway_router_is_none_without_keys(self):
        """LLMGateway.router is None when no API keys are provided."""
        with patch("app.llm.gateway.settings") as mock_settings, \
             patch("app.llm.gateway.Router"), \
             patch("app.llm.gateway.get_cohere_client") as mock_gc:

            mock_settings.google_api_key = ""
            mock_settings.openai_api_key = ""
            mock_settings.anthropic_api_key = ""
            mock_settings.groq_api_key = ""
            mock_settings.mistral_api_key = ""
            mock_settings.cohere_api_key = ""
            mock_gc.return_value = None

            from app.llm.gateway import LLMGateway
            gw = LLMGateway()
            assert gw.router is None

    @pytest.mark.asyncio
    @patch("app.llm.gateway.close_cohere_client", new_callable=AsyncMock)
    async def test_close_calls_close_cohere_client(self, mock_close):
        """LLMGateway.close() shuts down the Cohere client."""
        gw = _make_gateway()
        await gw.close()
        mock_close.assert_awaited_once()
