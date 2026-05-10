import json
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from main import get_chunks, retrieve_k_relevant_chunks, batch_process_questions


# ── Chunking ──────────────────────────────────────────────────────────────────

def test_get_chunks_exact_size(tmp_path):
    f = tmp_path / "kb.txt"
    f.write_text("abcdefghij")  # 10 chars
    chunks = get_chunks(str(f), chunk_size=5, overlap=0)
    assert chunks == ["abcde", "fghij"]


def test_get_chunks_with_overlap(tmp_path):
    f = tmp_path / "kb.txt"
    f.write_text("abcdefghij")
    chunks = get_chunks(str(f), chunk_size=6, overlap=2)
    # start=0 → "abcdef", start=4 → "efghij"
    assert chunks[0] == "abcdef"
    assert chunks[1] == "efghij"


def test_get_chunks_text_shorter_than_chunk(tmp_path):
    f = tmp_path / "kb.txt"
    f.write_text("hi")
    chunks = get_chunks(str(f), chunk_size=500, overlap=50)
    assert chunks == ["hi"]


def test_get_chunks_overlap_creates_continuity(tmp_path):
    f = tmp_path / "kb.txt"
    f.write_text("hello world")
    chunks = get_chunks(str(f), chunk_size=7, overlap=3)
    # Consecutive chunks share 3 characters — the tail of chunk[0] appears in chunk[1]
    assert chunks[0][-3:] == chunks[1][:3]


# ── Retrieval ─────────────────────────────────────────────────────────────────

def _make_db(vectors):
    """Build a fake database with provided vectors as embeddings."""
    return [{"text": f"chunk_{i}", "vector": v} for i, v in enumerate(vectors)]


def _mock_embedding(vector):
    embedding = MagicMock()
    embedding.embedding = vector
    response = MagicMock()
    response.data = [embedding]
    return response


def test_retrieve_returns_top_k_by_dot_product():
    # Query vector points in direction of chunk_1 (high dot product with [0,1])
    query_vector = [0.0, 1.0]
    db = _make_db([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])

    mock_instance = MagicMock()
    mock_instance.embeddings.create.return_value = _mock_embedding(query_vector)
    with patch("main.client", return_value=mock_instance):
        result = retrieve_k_relevant_chunks("any query", db, k=1)

    assert result == ["chunk_1"]


def test_retrieve_k_limits_results():
    query_vector = [1.0, 0.0]
    db = _make_db([[1.0, 0.0], [0.9, 0.0], [0.8, 0.0], [0.1, 0.0]])

    mock_instance = MagicMock()
    mock_instance.embeddings.create.return_value = _mock_embedding(query_vector)
    with patch("main.client", return_value=mock_instance):
        result = retrieve_k_relevant_chunks("query", db, k=2)

    assert len(result) == 2


# ── Batch processing uses the RAG pipeline ────────────────────────────────────

def test_batch_process_uses_rag_not_bare_gpt(tmp_path):
    kb = tmp_path / "kb.txt"
    kb.write_text("Formula One started in 1950.")
    questions = tmp_path / "q.txt"
    questions.write_text("When did Formula One start?")

    fake_embedding = MagicMock()
    fake_embedding.embedding = [1.0, 0.0]
    fake_embed_response = MagicMock()
    fake_embed_response.data = [fake_embedding]

    fake_message = MagicMock()
    fake_message.content = "Formula One started in 1950."
    fake_choice = MagicMock()
    fake_choice.message = fake_message
    fake_chat_response = MagicMock()
    fake_chat_response.choices = [fake_choice]

    mock_instance = MagicMock()
    mock_instance.embeddings.create.return_value = fake_embed_response
    mock_instance.chat.completions.create.return_value = fake_chat_response

    with patch("main.client", return_value=mock_instance):
        batch_process_questions(str(questions), str(kb))

        # embeddings.create must be called — once for the corpus, once per question
        assert mock_instance.embeddings.create.call_count >= 2

        # The chat call must include the retrieved context, not be a bare question
        call_args = mock_instance.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        user_message = next(m["content"] for m in messages if m["role"] == "user")
        assert "Context" in user_message or "context" in user_message
