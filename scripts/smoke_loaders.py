#!/usr/bin/env python3
"""
Smoke test script for dataset loaders.
Network access required - runs real dataset fetches.
Usage: python -m scripts.smoke_loaders
"""
import sys


def smoke_fineweb_edu():
    """Test fineweb-edu streaming loader."""
    print("\n=== FineWeb-Edu (streaming) ===")
    from niels_gpt.data.fineweb_edu import iter_fineweb_edu

    try:
        samples = list(iter_fineweb_edu(name="CC-MAIN-2024-10", take=2))
        print(f"Loaded {len(samples)} samples")
        for i, sample in enumerate(samples, 1):
            print(f"\nSample {i}:")
            print(f"  source: {sample.source}")
            print(f"  text (first 120 chars): {sample.text[:120]}...")
            print(f"  meta keys: {list(sample.meta.keys())}")
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    return True


def smoke_wikitext():
    """Test wikitext loader."""
    print("\n=== WikiText ===")
    from niels_gpt.data.wikitext import iter_wikitext

    try:
        samples = list(iter_wikitext(take=2))
        print(f"Loaded {len(samples)} samples")
        for i, sample in enumerate(samples, 1):
            print(f"\nSample {i}:")
            print(f"  source: {sample.source}")
            print(f"  text (first 120 chars): {sample.text[:120]}...")
            print(f"  meta: {sample.meta}")
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    return True


def smoke_dolly():
    """Test dolly loader."""
    print("\n=== Dolly-15k ===")
    from niels_gpt.data.dolly import iter_dolly_sft

    try:
        samples = list(iter_dolly_sft(take=2))
        print(f"Loaded {len(samples)} samples")
        for i, sample in enumerate(samples, 1):
            print(f"\nSample {i}:")
            print(f"  source: {sample.source}")
            print(f"  num messages: {len(sample.messages)}")
            for msg in sample.messages:
                content_preview = msg.content[:80].replace('\n', ' ')
                print(f"    {msg.role}: {content_preview}...")
            print(f"  meta keys: {list(sample.meta.keys())}")
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    return True


def smoke_oasst1():
    """Test oasst1 loader."""
    print("\n=== OASST1 ===")
    from niels_gpt.data.oasst1 import iter_oasst1_sft

    try:
        samples = list(iter_oasst1_sft(take_trees=2))
        print(f"Loaded {len(samples)} thread samples")
        for i, sample in enumerate(samples, 1):
            print(f"\nThread {i}:")
            print(f"  source: {sample.source}")
            print(f"  num messages: {len(sample.messages)}")
            print(f"  tree_id: {sample.meta['message_tree_id']}")
            for msg in sample.messages:
                content_preview = msg.content[:80].replace('\n', ' ')
                print(f"    {msg.role}: {content_preview}...")
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    return True


def smoke_gutenberg():
    """Test gutenberg loader."""
    print("\n=== Gutenberg Clean EN ===")
    from niels_gpt.data.gutenberg import iter_gutenberg

    try:
        samples = list(iter_gutenberg(take=2))
        print(f"Loaded {len(samples)} samples")
        for i, sample in enumerate(samples, 1):
            print(f"\nSample {i}:")
            print(f"  source: {sample.source}")
            print(f"  text (first 120 chars): {sample.text[:120]}...")
            print(f"  meta keys: {list(sample.meta.keys())}")
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    return True


def main():
    """Run all smoke tests."""
    print("Starting dataset loader smoke tests...")
    print("(Network access required)")

    results = {
        "fineweb-edu": smoke_fineweb_edu(),
        "wikitext": smoke_wikitext(),
        "dolly": smoke_dolly(),
        "oasst1": smoke_oasst1(),
        "gutenberg": smoke_gutenberg(),
    }

    print("\n" + "=" * 60)
    print("RESULTS:")
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {name}: {status}")

    if all(results.values()):
        print("\nAll smoke tests passed!")
        return 0
    else:
        print("\nSome smoke tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
