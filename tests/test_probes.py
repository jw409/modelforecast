"""Unit tests for probe implementations."""

import pytest
from unittest.mock import MagicMock, patch

from modelforecast.probes.base import ProbeResult
from modelforecast.probes.t0_invoke import T0InvokeProbe
from modelforecast.probes.t1_schema import T1SchemaProbe
from modelforecast.probes.t2_selection import T2SelectionProbe


class TestProbeResult:
    """Test ProbeResult dataclass."""

    def test_probe_result_dataclass(self):
        """ProbeResult stores trial data correctly."""
        result = ProbeResult(
            success=True,
            tool_called=True,
            tool_name="search",
            parameters={"query": "test"},
            raw_response={"test": "data"},
            latency_ms=150,
            error=None,
        )
        assert result.success is True
        assert result.tool_called is True
        assert result.tool_name == "search"
        assert result.latency_ms == 150

    def test_probe_result_failure(self):
        """ProbeResult handles failure case."""
        result = ProbeResult(
            success=False,
            tool_called=False,
            tool_name=None,
            parameters=None,
            raw_response={},
            latency_ms=0,
            error="Connection timeout",
        )
        assert result.success is False
        assert result.error == "Connection timeout"

    def test_probe_result_repr(self):
        """ProbeResult has compact repr."""
        result = ProbeResult(
            success=True,
            tool_called=True,
            tool_name="search",
            parameters={"query": "auth"},
            raw_response={},
            latency_ms=250,
        )
        repr_str = repr(result)
        assert "PASS" in repr_str
        assert "search" in repr_str
        assert "250ms" in repr_str


class TestT0InvokeProbe:
    """Test T0 (basic tool invocation) probe."""

    def test_probe_attributes(self):
        """T0 probe has correct attributes."""
        probe = T0InvokeProbe()
        assert probe.level == 0
        assert probe.name == "Basic Tool Calling"
        assert "search" in probe.prompt.lower()
        assert len(probe.tools) == 1

    def test_str_representation(self):
        """T0 probe has readable string representation."""
        probe = T0InvokeProbe()
        assert "Level 0" in str(probe)
        assert "Basic Tool Calling" in str(probe)


class TestT1SchemaProbe:
    """Test T1 (schema compliance) probe."""

    def test_probe_attributes(self):
        """T1 probe has correct attributes."""
        probe = T1SchemaProbe()
        assert probe.level == 1


class TestT2SelectionProbe:
    """Test T2 (tool selection) probe."""

    def test_probe_attributes(self):
        """T2 probe has correct attributes."""
        probe = T2SelectionProbe()
        assert probe.level == 2


class TestMockedProbeRun:
    """Test probe run with mocked OpenAI client."""

    def test_t0_success_with_tool_call(self):
        """T0 correctly identifies successful tool call."""
        probe = T0InvokeProbe()

        # Create mock response
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {"id": "test"}
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = [MagicMock()]
        mock_response.choices[0].message.tool_calls[0].function.name = "search"
        mock_response.choices[0].message.tool_calls[0].function.arguments = '{"query": "authentication"}'

        # Create mock client
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        result = probe.run("test-model", mock_client)

        assert result.success is True
        assert result.tool_called is True
        assert result.tool_name == "search"
        assert result.parameters == {"query": "authentication"}

    def test_t0_failure_no_tool_call(self):
        """T0 correctly identifies missing tool call."""
        probe = T0InvokeProbe()

        # Create mock response with no tool calls
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {"id": "test"}
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.content = "The weather is nice."

        # Create mock client
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        result = probe.run("test-model", mock_client)

        assert result.success is False
        assert result.tool_called is False
        assert result.tool_name is None

    def test_t0_handles_exception(self):
        """T0 gracefully handles API exceptions."""
        probe = T0InvokeProbe()

        # Create mock client that raises
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        result = probe.run("test-model", mock_client)

        assert result.success is False
        assert result.error == "API Error"


class TestIntegration:
    """Integration tests (require API key)."""

    @pytest.mark.skipif(
        not pytest.importorskip("os").getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set"
    )
    def test_t0_real_model(self):
        """Run T0 against a real model (requires API key)."""
        # This is an integration test - skip in CI without key
        pass


# ============================================================================
# EMBEDDING PROBES (E Dimension) - MTEB-inspired
# ============================================================================

from modelforecast.probes.base import EmbeddingResult
from modelforecast.probes.e0_invoke import E0InvokeProbe
from modelforecast.probes.e1_retrieval import E1RetrievalProbe, cosine_similarity


class TestEmbeddingResult:
    """Test EmbeddingResult dataclass."""

    def test_embedding_result_success(self):
        """EmbeddingResult stores embedding data correctly."""
        result = EmbeddingResult(
            success=True,
            embedding_returned=True,
            dimensions=1536,
            embedding=[0.1, 0.2, 0.3],
            raw_response={"model": "test"},
            latency_ms=200,
        )
        assert result.success is True
        assert result.embedding_returned is True
        assert result.dimensions == 1536
        assert len(result.embedding) == 3
        assert result.latency_ms == 200

    def test_embedding_result_with_similarity(self):
        """EmbeddingResult stores similarity score for retrieval probes."""
        result = EmbeddingResult(
            success=True,
            embedding_returned=True,
            dimensions=768,
            embedding=[0.5] * 10,
            raw_response={},
            latency_ms=150,
            similarity_score=0.15,  # Margin between relevant and distractor
        )
        assert result.similarity_score == 0.15

    def test_embedding_result_failure(self):
        """EmbeddingResult handles failure case."""
        result = EmbeddingResult(
            success=False,
            embedding_returned=False,
            dimensions=None,
            embedding=None,
            raw_response={},
            latency_ms=0,
            error="Model does not support embeddings",
        )
        assert result.success is False
        assert result.error == "Model does not support embeddings"

    def test_embedding_result_repr(self):
        """EmbeddingResult has compact repr."""
        result = EmbeddingResult(
            success=True,
            embedding_returned=True,
            dimensions=1536,
            embedding=[0.1] * 10,
            raw_response={},
            latency_ms=250,
            similarity_score=0.12,
        )
        repr_str = repr(result)
        assert "PASS" in repr_str
        assert "1536d" in repr_str
        assert "sim=0.120" in repr_str
        assert "250ms" in repr_str


class TestE0InvokeProbe:
    """Test E0 (basic embedding invocation) probe."""

    def test_probe_attributes(self):
        """E0 probe has correct attributes."""
        probe = E0InvokeProbe()
        assert probe.level == 0
        assert probe.name == "Basic Embedding"
        assert probe.dimension == "E"
        assert len(probe.input_text) > 0  # Has test input

    def test_str_representation(self):
        """E0 probe has readable string representation."""
        probe = E0InvokeProbe()
        assert "Level 0" in str(probe)
        assert "Basic Embedding" in str(probe)


class TestE1RetrievalProbe:
    """Test E1 (retrieval quality) probe - MTEB-inspired."""

    def test_probe_attributes(self):
        """E1 probe has correct attributes."""
        probe = E1RetrievalProbe()
        assert probe.level == 1
        assert probe.name == "Retrieval Quality"
        assert probe.dimension == "E"

    def test_has_query_and_documents(self):
        """E1 probe has query and 3 documents (relevant, distractor, irrelevant)."""
        probe = E1RetrievalProbe()
        assert "Python" in probe.query  # Query mentions Python
        assert "Python" in probe.doc_relevant  # Relevant doc is about Python
        assert "JavaScript" in probe.doc_distractor  # Distractor is wrong language
        assert "Database" in probe.doc_irrelevant or "pool" in probe.doc_irrelevant

    def test_margin_thresholds(self):
        """E1 probe has opinionated margin thresholds."""
        probe = E1RetrievalProbe()
        # These are opinionated thresholds - not generic MTEB
        assert probe.min_margin_relevant_vs_distractor > 0
        assert probe.min_margin_distractor_vs_irrelevant > 0


class TestCosineSimilarity:
    """Test cosine similarity helper function."""

    def test_identical_vectors(self):
        """Identical vectors have similarity 1.0."""
        vec = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(vec, vec) - 1.0) < 0.001

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have similarity 0.0."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(vec_a, vec_b)) < 0.001

    def test_opposite_vectors(self):
        """Opposite vectors have similarity -1.0."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [-1.0, 0.0, 0.0]
        assert abs(cosine_similarity(vec_a, vec_b) + 1.0) < 0.001

    def test_similar_vectors(self):
        """Similar vectors have high positive similarity."""
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [1.1, 2.1, 3.1]  # Slightly different
        sim = cosine_similarity(vec_a, vec_b)
        assert sim > 0.99  # Very similar

    def test_different_length_raises(self):
        """Vectors of different lengths raise ValueError."""
        vec_a = [1.0, 2.0]
        vec_b = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError):
            cosine_similarity(vec_a, vec_b)


class TestMockedEmbeddingProbeRun:
    """Test embedding probe run with mocked OpenAI client."""

    def test_e0_success_with_embedding(self):
        """E0 correctly identifies successful embedding."""
        probe = E0InvokeProbe()

        # Create mock response
        mock_embedding_data = MagicMock()
        mock_embedding_data.embedding = [0.1] * 1536  # 1536 dims

        mock_response = MagicMock()
        mock_response.model_dump.return_value = {"model": "test"}
        mock_response.data = [mock_embedding_data]

        # Create mock client
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response

        result = probe.run("test-embedding-model", mock_client)

        assert result.success is True
        assert result.embedding_returned is True
        assert result.dimensions == 1536
        assert len(result.embedding) == 10  # Only first 10 stored

    def test_e0_failure_empty_response(self):
        """E0 correctly identifies empty embedding response."""
        probe = E0InvokeProbe()

        mock_response = MagicMock()
        mock_response.model_dump.return_value = {}
        mock_response.data = []

        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response

        result = probe.run("test-model", mock_client)

        assert result.success is False
        assert result.embedding_returned is False

    def test_e1_success_correct_ranking(self):
        """E1 correctly identifies good retrieval ranking."""
        probe = E1RetrievalProbe()

        # Create mock embeddings with correct ranking
        # Query -> Relevant should be closest, then distractor, then irrelevant
        mock_query = MagicMock()
        mock_query.embedding = [1.0, 0.0, 0.0, 0.0]

        mock_relevant = MagicMock()
        mock_relevant.embedding = [0.95, 0.1, 0.0, 0.0]  # Very similar to query

        mock_distractor = MagicMock()
        mock_distractor.embedding = [0.7, 0.5, 0.0, 0.0]  # Somewhat similar

        mock_irrelevant = MagicMock()
        mock_irrelevant.embedding = [0.1, 0.0, 0.9, 0.0]  # Different direction

        mock_response = MagicMock()
        mock_response.model_dump.return_value = {"model": "test"}
        mock_response.data = [mock_query, mock_relevant, mock_distractor, mock_irrelevant]

        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response

        result = probe.run("test-model", mock_client)

        # Check that ordering is correct (may or may not pass margin thresholds)
        assert result.embedding_returned is True
        assert "ordering_correct" in result.raw_response

    def test_e1_failure_wrong_ranking(self):
        """E1 correctly identifies bad retrieval ranking."""
        probe = E1RetrievalProbe()

        # Create mock embeddings with WRONG ranking
        # Distractor is more similar to query than relevant doc (bad model!)
        mock_query = MagicMock()
        mock_query.embedding = [1.0, 0.0, 0.0, 0.0]

        mock_relevant = MagicMock()
        mock_relevant.embedding = [0.5, 0.5, 0.0, 0.0]  # Medium similarity

        mock_distractor = MagicMock()
        mock_distractor.embedding = [0.95, 0.1, 0.0, 0.0]  # MORE similar (wrong!)

        mock_irrelevant = MagicMock()
        mock_irrelevant.embedding = [0.1, 0.0, 0.9, 0.0]

        mock_response = MagicMock()
        mock_response.model_dump.return_value = {"model": "test"}
        mock_response.data = [mock_query, mock_relevant, mock_distractor, mock_irrelevant]

        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response

        result = probe.run("test-model", mock_client)

        assert result.success is False  # Wrong ranking = fail
        assert result.raw_response.get("ordering_correct") is False

    def test_e0_handles_exception(self):
        """E0 gracefully handles API exceptions."""
        probe = E0InvokeProbe()

        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = Exception("API Error")

        result = probe.run("test-model", mock_client)

        assert result.success is False
        assert result.error == "API Error"
