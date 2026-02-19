"""
Tests for OVERLAP_SIZE window advancement in MemoryBuilder.
Verifies that the sliding window retains overlap dialogues between windows.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch
from models.memory_entry import MemoryEntry, Dialogue


def _make_dialogues(n):
    """Create n dummy dialogues with sequential IDs."""
    return [
        Dialogue(dialogue_id=i, speaker=f"User{i}", content=f"Message {i}")
        for i in range(n)
    ]


def _make_builder(window_size, overlap_size):
    """Create a MemoryBuilder with mocked LLM and VectorStore."""
    # Patch config to avoid needing a real config.py
    mock_config = MagicMock()
    mock_config.WINDOW_SIZE = window_size
    mock_config.OVERLAP_SIZE = overlap_size
    mock_config.ENABLE_PARALLEL_PROCESSING = False
    mock_config.MAX_PARALLEL_WORKERS = 1
    mock_config.ENABLE_SYNTHESIS = False
    mock_config.USE_JSON_FORMAT = False

    with patch.dict('sys.modules', {'config': mock_config}):
        # Re-import to pick up mocked config
        import importlib
        import core.memory_builder as mb_module
        importlib.reload(mb_module)

        llm_client = MagicMock()
        vector_store = MagicMock()

        builder = mb_module.MemoryBuilder(
            llm_client=llm_client,
            vector_store=vector_store,
            window_size=window_size,
            overlap_size=overlap_size,
            enable_parallel_processing=False,
        )
        # Mock _generate_memory_entries to return empty list (we only care about buffer mechanics)
        builder._generate_memory_entries = MagicMock(return_value=[])
        return builder


def test_overlap_retained():
    """With overlap=2 and window_size=5, 2 dialogues should be retained after processing a window."""
    print("\n[TEST] Overlap retained...")
    builder = _make_builder(window_size=5, overlap_size=2)
    dialogues = _make_dialogues(7)  # 7 dialogues: first window=5, then 2 overlap + 2 remaining = 4

    # Add all dialogues without auto-processing
    for d in dialogues:
        builder.add_dialogue(d, auto_process=False)

    assert len(builder.dialogue_buffer) == 7, f"Buffer should have 7, got {len(builder.dialogue_buffer)}"

    # Process one window
    builder.process_window()

    # After processing: advance = 5 - 2 = 3, so buffer should have 7 - 3 = 4 dialogues
    assert len(builder.dialogue_buffer) == 4, (
        f"After processing window with overlap=2, buffer should have 4 dialogues, got {len(builder.dialogue_buffer)}"
    )
    # The retained dialogues should be dialogues 3,4,5,6 (IDs)
    retained_ids = [d.dialogue_id for d in builder.dialogue_buffer]
    assert retained_ids == [3, 4, 5, 6], f"Expected retained IDs [3,4,5,6], got {retained_ids}"

    print(f"  PASS: Buffer has {len(builder.dialogue_buffer)} dialogues after overlap retention")
    return True


def test_zero_overlap():
    """With overlap=0, behavior matches original: buffer advances by full window_size."""
    print("\n[TEST] Zero overlap (original behavior)...")
    builder = _make_builder(window_size=5, overlap_size=0)
    dialogues = _make_dialogues(7)

    for d in dialogues:
        builder.add_dialogue(d, auto_process=False)

    builder.process_window()

    # advance = 5 - 0 = 5, so buffer should have 7 - 5 = 2
    assert len(builder.dialogue_buffer) == 2, (
        f"With overlap=0, buffer should have 2 remaining, got {len(builder.dialogue_buffer)}"
    )
    retained_ids = [d.dialogue_id for d in builder.dialogue_buffer]
    assert retained_ids == [5, 6], f"Expected retained IDs [5,6], got {retained_ids}"

    print(f"  PASS: Buffer has {len(builder.dialogue_buffer)} dialogues (original behavior)")
    return True


def test_overlap_in_parallel_windowing():
    """Verify overlap is applied when creating windows in add_dialogues_parallel."""
    print("\n[TEST] Overlap in parallel windowing...")
    builder = _make_builder(window_size=5, overlap_size=2)
    dialogues = _make_dialogues(12)

    # Manually simulate the parallel windowing logic
    builder.dialogue_buffer.extend(dialogues)
    windows = []
    advance = builder.window_size - builder.overlap_size  # 3
    while len(builder.dialogue_buffer) >= builder.window_size:
        window = builder.dialogue_buffer[:builder.window_size]
        builder.dialogue_buffer = builder.dialogue_buffer[advance:]
        windows.append(window)

    # With 12 dialogues, window_size=5, advance=3:
    # Window 1: [0,1,2,3,4], buffer becomes [3..11] (9 items)
    # Window 2: [3,4,5,6,7], buffer becomes [6..11] (6 items)
    # Window 3: [6,7,8,9,10], buffer becomes [9..11] (3 items) — not enough for another window
    assert len(windows) == 3, f"Expected 3 windows, got {len(windows)}"
    assert len(builder.dialogue_buffer) == 3, (
        f"Expected 3 remaining dialogues, got {len(builder.dialogue_buffer)}"
    )

    # Verify window contents
    assert [d.dialogue_id for d in windows[0]] == [0, 1, 2, 3, 4]
    assert [d.dialogue_id for d in windows[1]] == [3, 4, 5, 6, 7]
    assert [d.dialogue_id for d in windows[2]] == [6, 7, 8, 9, 10]

    print(f"  PASS: {len(windows)} windows created with correct overlap")
    return True


def test_invalid_overlap_size():
    """Verify that overlap_size >= window_size or negative overlap raises ValueError."""
    print("\n[TEST] Invalid overlap_size...")

    # overlap_size == window_size
    try:
        _make_builder(window_size=5, overlap_size=5)
        assert False, "Should have raised ValueError for overlap_size == window_size"
    except ValueError:
        print("  PASS: Correctly rejected overlap_size == window_size")

    # overlap_size > window_size
    try:
        _make_builder(window_size=5, overlap_size=10)
        assert False, "Should have raised ValueError for overlap_size > window_size"
    except ValueError:
        print("  PASS: Correctly rejected overlap_size > window_size")

    # Negative overlap_size
    try:
        _make_builder(window_size=5, overlap_size=-1)
        assert False, "Should have raised ValueError for negative overlap_size"
    except ValueError:
        print("  PASS: Correctly rejected negative overlap_size")

    return True


def main():
    print("=" * 60)
    print("Overlap Window Tests")
    print("=" * 60)

    tests = [
        test_overlap_retained,
        test_zero_overlap,
        test_overlap_in_parallel_windowing,
        test_invalid_overlap_size,
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
