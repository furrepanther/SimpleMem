"""
Tests for SemanticSynthesizer (Stage 2: Online Semantic Synthesis).
Uses mocked dependencies to test synthesis logic without LLM or VectorStore.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np

from models.memory_entry import MemoryEntry


def _make_entry(text="Test entry", entry_id=None, **kwargs):
    """Create a MemoryEntry with defaults."""
    params = dict(
        lossless_restatement=text,
        keywords=kwargs.get("keywords", ["test"]),
        persons=kwargs.get("persons", []),
        entities=kwargs.get("entities", []),
    )
    if entry_id:
        params["entry_id"] = entry_id
    return MemoryEntry(**params)


def _make_synthesizer(find_similar_return=None, merge_response=None, enable=True):
    """Create a SemanticSynthesizer with mocked dependencies."""
    mock_config = MagicMock()
    mock_config.ENABLE_SYNTHESIS = enable
    mock_config.SYNTHESIS_SIMILARITY_THRESHOLD = 0.85
    mock_config.SYNTHESIS_MAX_CANDIDATES = 10
    mock_config.USE_JSON_FORMAT = False

    with patch.dict('sys.modules', {'config': mock_config}):
        import importlib
        import core.synthesis as syn_module
        importlib.reload(syn_module)

        llm_client = MagicMock()
        vector_store = MagicMock()
        embedding_model = MagicMock()

        # Mock embedding: returns unit vectors
        embedding_model.encode_documents = MagicMock(
            return_value=np.array([[1.0, 0.0, 0.0]] * 10)[:1]  # will be sliced per call
        )

        def _encode_docs(texts):
            return np.array([[1.0, 0.0, 0.0]] * len(texts))

        embedding_model.encode_documents = MagicMock(side_effect=_encode_docs)

        # Mock find_similar_entries
        vector_store.find_similar_entries = MagicMock(
            return_value=find_similar_return or []
        )
        vector_store.embedding_model = embedding_model

        # Mock LLM merge response
        if merge_response:
            llm_client.chat_completion = MagicMock(return_value=merge_response)
            llm_client.extract_json = MagicMock(return_value=merge_response)
        else:
            llm_client.chat_completion = MagicMock(return_value='{}')
            llm_client.extract_json = MagicMock(return_value={})

        synthesizer = syn_module.SemanticSynthesizer(
            llm_client=llm_client,
            vector_store=vector_store,
            embedding_model=embedding_model,
            similarity_threshold=0.85,
            max_candidates=10,
        )

        return synthesizer, llm_client, vector_store


def test_empty_entries():
    """Empty input should return empty result."""
    print("\n[TEST] Empty entries...")
    synth, _, _ = _make_synthesizer()
    result = synth.synthesize([])
    assert len(result.entries_to_store) == 0
    assert len(result.entries_to_delete) == 0
    assert result.merged_count == 0
    assert result.passthrough_count == 0
    print("  PASS: Empty input returns empty result")
    return True


def test_no_similar_passthrough():
    """When no similar entries exist, all entries pass through unchanged."""
    print("\n[TEST] No similar entries passthrough...")
    synth, _, vs = _make_synthesizer(find_similar_return=[])

    entries = [_make_entry("Entry A"), _make_entry("Entry B")]
    result = synth.synthesize(entries)

    assert len(result.entries_to_store) == 2, f"Expected 2 entries, got {len(result.entries_to_store)}"
    assert result.passthrough_count == 2
    assert result.merged_count == 0
    assert len(result.entries_to_delete) == 0
    print(f"  PASS: {result.passthrough_count} entries passed through")
    return True


def test_similar_triggers_merge():
    """When a similar entry is found, LLM merge is called and old entry is scheduled for deletion."""
    print("\n[TEST] Similar entry triggers merge...")
    existing = _make_entry("Existing entry about meeting", entry_id="existing-123")

    merge_data = {
        "lossless_restatement": "Merged: meeting details combined",
        "keywords": ["meeting", "details"],
        "timestamp": None,
        "location": None,
        "persons": [],
        "entities": [],
        "topic": "meeting",
    }

    similar_result = [{
        "entry": existing,
        "vector": [1.0, 0.0, 0.0],
        "cosine_similarity": 0.92,
    }]

    synth, llm, vs = _make_synthesizer(
        find_similar_return=similar_result,
        merge_response=merge_data,
    )

    new_entry = _make_entry("New entry about meeting")
    result = synth.synthesize([new_entry])

    assert result.merged_count == 1, f"Expected 1 merge, got {result.merged_count}"
    assert len(result.entries_to_delete) == 1, f"Expected 1 deletion, got {len(result.entries_to_delete)}"
    assert result.entries_to_delete[0] == "existing-123"
    assert len(result.entries_to_store) == 1
    # Merged entry should reuse existing entry_id
    assert result.entries_to_store[0].entry_id == "existing-123"
    assert "Merged" in result.entries_to_store[0].lossless_restatement

    print("  PASS: Merge triggered, old entry scheduled for deletion")
    return True


def test_merge_failure_stores_new():
    """When LLM merge fails, new entry is stored as-is with no deletion."""
    print("\n[TEST] Merge failure stores new entry...")
    existing = _make_entry("Existing entry", entry_id="existing-456")

    similar_result = [{
        "entry": existing,
        "vector": [1.0, 0.0, 0.0],
        "cosine_similarity": 0.90,
    }]

    synth, llm, vs = _make_synthesizer(find_similar_return=similar_result)
    # Make LLM calls raise exceptions to simulate failure
    llm.chat_completion = MagicMock(side_effect=Exception("LLM API error"))

    new_entry = _make_entry("New entry that can't be merged")
    result = synth.synthesize([new_entry])

    assert result.merged_count == 0, f"Expected 0 merges, got {result.merged_count}"
    assert result.passthrough_count == 1, f"Expected 1 passthrough, got {result.passthrough_count}"
    assert len(result.entries_to_delete) == 0
    assert len(result.entries_to_store) == 1
    # Should be the original new entry, not merged
    assert result.entries_to_store[0].lossless_restatement == "New entry that can't be merged"

    print("  PASS: Failed merge stores new entry as-is")
    return True


def main():
    print("=" * 60)
    print("Semantic Synthesis Tests")
    print("=" * 60)

    tests = [
        test_empty_entries,
        test_no_similar_passthrough,
        test_similar_triggers_merge,
        test_merge_failure_stores_new,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
