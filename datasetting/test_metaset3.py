from datasetting import load_metaset3_dataset

def test_metaset3_loading():
    """Test that metaset3 data loads and transforms correctly"""
    print("Testing metaset3 dataset loading...")

    dataset = load_metaset3_dataset("train")
    print(f"Loaded {len(dataset)} examples")

    example = dataset[0]
    print("\nFirst example structure:")
    print(f"Keys: {list(example.keys())}")

    messages = example["messages"]
    print(f"\nMessages type: {type(messages)}")
    print(f"First message keys: {list(messages[0].keys())}")

    content = messages[0]["content"][0]["text"]
    print("\nPrompt preview (first 200 chars):")
    print(content[:200] + "...")

    assert "Input 4:" in content
    assert "Output 4:" in content

    answer = example["answer"]
    print(f"\nAnswer: {answer}")

    print("\n✓ Dataset loading test passed!")

def test_transform_function():
    """Test the transform function with a sample metaset3 item"""
    print("\nTesting transform function...")

    sample_item = {
        "tests": [
            {"output": 0, "input": {"a": 5}},
            {"output": 1, "input": {"a": 20}},
            {"output": 2, "input": {"a": 28}}
        ],
        "short_tree": ["invoke1", ["lambda1", ["if", ["==", ["len", ["digits", "arg1"]], "1"], "0", ["+", "1", ["self", ["reduce", ["digits", "arg1"], "0", "+"]]]]], "a"],
        "text": ["given", "a", "number", "a", ",", "find", "how", "many", "times"],
        "args": {"a": "int"},
        "return_type": "int"
    }
