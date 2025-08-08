from cosmos_reason1_utils.text import extract_text


def test_extract_text():
    s1 = "This is a test"
    s2 = "This is an answer"
    # Empty match
    assert extract_text("<think></think>", "think") == ""
    # Empty text
    assert extract_text("", "think") is None
    # Empty key
    assert extract_text(f"<>{s1}</>", "") == s1
    # Basic
    assert extract_text(f"<think>{s1}</think>", "think") == s1
    # Wrong key
    assert extract_text(f"<think>{s1}</think>", "answer") is None
    # No closing tag
    assert extract_text(f"</think>{s1}<think>", "think") is None
    # Other text
    assert extract_text(f"<think>{s1}</think>{s1}", "think") == s1
    # Other keys
    assert extract_text(f"<think>{s1}</think><answer>{s2}</answer>", "answer") == s2
    assert extract_text(f"<think>{s1}</think><answer>{s2}</answer>", "think") == s1
    # Multiple matches
    assert extract_text(f"<think>{s1}</think><think>{s1}</think>", "think") is None
    assert extract_text(f"<think>{s1}</think><think>{s2}</think>", "think") is None
