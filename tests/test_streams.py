"""Tests for streams module."""

import json
from pathlib import Path

import pytest

from niels_gpt.streams import (
    StreamBuildConfig,
    build_primer_stream,
    build_roam_stream,
    build_sources,
    build_wiki_stream,
)


@pytest.fixture(autouse=True)
def block_wiki_download(monkeypatch):
    """Fail fast if wiki is ever touched in tests."""
    def _raise(*args, **kwargs):
        raise RuntimeError("load_wikitext should not be called in tests")

    monkeypatch.setattr("niels_gpt.streams.load_wikitext", _raise)
    yield


class TestStreamBuilders:
    """Test individual stream builder functions."""

    def test_build_wiki_stream(self):
        """Test wiki stream building with separator."""
        docs = ["doc1", "doc2", "doc3"]
        sep = "\n\n"
        result = build_wiki_stream(docs, sep=sep)
        assert result == b"doc1\n\ndoc2\n\ndoc3"

    def test_build_roam_stream(self):
        """Test roam stream building with separator."""
        docs = ["roam1", "roam2"]
        sep = "\n\n"
        result = build_roam_stream(docs, sep=sep)
        assert result == b"roam1\n\nroam2"

    def test_build_primer_stream(self):
        """Test primer stream encoding as-is."""
        text = "system: hello\nuser: hi\nassistant: hey"
        result = build_primer_stream(text)
        assert result == text.encode("utf-8")


class TestBuildSources:
    """Test build_sources with caching."""

    def test_default_config_matches_default_p_train_keys(self):
        """Default stream config should align with default p_train keys."""
        cfg = StreamBuildConfig()
        p_keys = {"wiki", "roam", "primer"}
        assert set(cfg.enabled_sources) == p_keys
        assert set(cfg.required_sources) == p_keys

    def test_build_sources_with_fake_data(self, tmp_path):
        """Test build_sources with fake roam and primer data."""
        # Create fake roam directory
        roam_dir = tmp_path / "roam"
        roam_dir.mkdir()
        (roam_dir / "file1.md").write_text("roam content 1" * 500)
        (roam_dir / "file2.md").write_text("roam content 2" * 100)
        (roam_dir / "file3.md").write_text("roam content 3" * 100)

        # Create fake primer file: 20 blocks, each 5k chars, joined properly
        primer_file = tmp_path / "primer.txt"
        big_dialogue = "system: " + ("x" * 5000) + "\nuser: y\nassistant: z\n"
        blocks = [big_dialogue for _ in range(20)]
        from niels_gpt.data import DIALOGUE_DELIM
        primer_content = DIALOGUE_DELIM.join(blocks)
        primer_file.write_text(primer_content)

        # Cache directory
        cache_dir = tmp_path / "cache"

        # Build config - skip wiki since we don't want to download it in tests
        cfg = StreamBuildConfig(
            roam_root=str(roam_dir),
            primer_path=str(primer_file),
            cache_dir=str(cache_dir),
            seed=42,
            required_sources=("roam", "primer"),  # Don't require wiki
            enabled_sources=("roam", "primer"),
        )

        # Build sources
        sources_train, sources_val = build_sources(cfg)

        # Check that roam and primer are in the results
        assert "roam" in sources_train
        assert "roam" in sources_val
        assert "primer" in sources_train
        assert "primer" in sources_val

        # Check that cache files were created
        assert cache_dir.exists()
        cache_files = list(cache_dir.glob("*"))
        assert len(cache_files) > 0

        # Check .bin and .meta.json files exist for at least one source
        bin_files = list(cache_dir.glob("*.bin"))
        meta_files = list(cache_dir.glob("*.meta.json"))
        assert len(bin_files) > 0
        assert len(meta_files) > 0

    def test_build_sources_cache_reuse(self, tmp_path):
        """Test that cache is reused on second call."""
        # Create fake roam directory with multiple files to ensure both train and val splits
        roam_dir = tmp_path / "roam"
        roam_dir.mkdir()
        (roam_dir / "file1.md").write_text("roam content 1" * 500)
        (roam_dir / "file2.md").write_text("roam content 2" * 500)
        (roam_dir / "file3.md").write_text("roam content 3" * 500)

        # Cache directory
        cache_dir = tmp_path / "cache"

        # Build config - skip primer to keep test simple
        cfg = StreamBuildConfig(
            roam_root=str(roam_dir),
            primer_path="nonexistent",  # Don't use primer
            cache_dir=str(cache_dir),
            seed=42,
            required_sources=("roam",),
            enabled_sources=("roam",),
        )

        # Build sources first time
        sources_train_1, sources_val_1 = build_sources(cfg)

        # Get mtime of cache files
        cache_files_before = {f: f.stat().st_mtime_ns for f in cache_dir.glob("*.bin")}

        # Build sources second time (should use cache)
        sources_train_2, sources_val_2 = build_sources(cfg)

        # Get mtime of cache files after
        cache_files_after = {f: f.stat().st_mtime_ns for f in cache_dir.glob("*.bin")}

        # Cache files should have same mtime (not rewritten)
        assert cache_files_before == cache_files_after

        # Results should be identical
        assert sources_train_1.keys() == sources_train_2.keys()
        assert sources_val_1.keys() == sources_val_2.keys()
        for key in sources_train_1.keys():
            assert sources_train_1[key] == sources_train_2[key]

    def test_build_sources_cache_invalidation_on_file_change(self, tmp_path):
        """Test that cache is invalidated when roam file changes."""
        # Create fake roam directory with multiple files to ensure both train and val splits
        roam_dir = tmp_path / "roam"
        roam_dir.mkdir()
        roam_file = roam_dir / "file1.md"
        roam_file.write_text("roam content 1" * 500)
        (roam_dir / "file2.md").write_text("roam content 2" * 500)
        (roam_dir / "file3.md").write_text("roam content 3" * 500)

        # Cache directory
        cache_dir = tmp_path / "cache"

        # Build config
        cfg = StreamBuildConfig(
            roam_root=str(roam_dir),
            primer_path="nonexistent",  # Don't use primer for this test
            cache_dir=str(cache_dir),
            seed=42,
            required_sources=("roam",),  # Only require roam
            enabled_sources=("roam",),
        )

        # Build sources first time
        sources_train_1, sources_val_1 = build_sources(cfg)

        # Modify roam file
        import time

        time.sleep(0.01)  # Ensure mtime changes
        roam_file.write_text("roam content MODIFIED" * 500)

        # Build sources second time (should rebuild)
        sources_train_2, sources_val_2 = build_sources(cfg)

        # Results should be different
        if "roam" in sources_train_1 and "roam" in sources_train_2:
            assert sources_train_1["roam"] != sources_train_2["roam"]

    def test_build_sources_length_guard(self, tmp_path):
        """Test that build_sources raises if streams are too short."""
        # Create fake roam directory with very short content (multiple files to ensure val split)
        roam_dir = tmp_path / "roam"
        roam_dir.mkdir()
        (roam_dir / "file1.md").write_text("x")  # Only 1 byte
        (roam_dir / "file2.md").write_text("y")  # Only 1 byte

        # Cache directory
        cache_dir = tmp_path / "cache"

        # Build config
        cfg = StreamBuildConfig(
            roam_root=str(roam_dir),
            primer_path="nonexistent",
            cache_dir=str(cache_dir),
            seed=42,
            required_sources=("roam",),
            enabled_sources=("roam",),
        )

        # Build sources should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            build_sources(cfg)

        # Check error message mentions the short stream
        assert "too short" in str(exc_info.value).lower()
        assert "257" in str(exc_info.value) or "T=" in str(exc_info.value)

    def test_metadata_format(self, tmp_path):
        """Test that metadata files have correct format."""
        # Create fake roam directory with multiple files to ensure both train and val splits
        roam_dir = tmp_path / "roam"
        roam_dir.mkdir()
        (roam_dir / "file1.md").write_text("roam content 1" * 500)
        (roam_dir / "file2.md").write_text("roam content 2" * 500)
        (roam_dir / "file3.md").write_text("roam content 3" * 500)

        # Create fake primer file: 20 blocks, each 5k chars
        primer_file = tmp_path / "primer.txt"
        big_dialogue = "system: " + ("x" * 5000) + "\nuser: y\nassistant: z\n"
        blocks = [big_dialogue for _ in range(20)]
        from niels_gpt.data import DIALOGUE_DELIM
        primer_content = DIALOGUE_DELIM.join(blocks)
        primer_file.write_text(primer_content)

        # Cache directory
        cache_dir = tmp_path / "cache"

        # Build config
        cfg = StreamBuildConfig(
            roam_root=str(roam_dir),
            primer_path=str(primer_file),
            cache_dir=str(cache_dir),
            seed=42,
            required_sources=("roam", "primer"),
            enabled_sources=("roam", "primer"),
        )

        # Build sources
        sources_train, sources_val = build_sources(cfg)

        # Check metadata files
        for meta_file in cache_dir.glob("*.meta.json"):
            with open(meta_file, "r") as f:
                metadata = json.load(f)

            # All metadata should have source and split
            assert "source" in metadata
            assert "split" in metadata

            source = metadata["source"]

            if source == "roam":
                assert "files" in metadata
                assert "sep" in metadata
                assert "seed" in metadata
                assert "val_frac" in metadata
                # Each file should have path, mtime_ns, size
                for file_meta in metadata["files"]:
                    assert "path" in file_meta
                    assert "mtime_ns" in file_meta
                    assert "size" in file_meta
            elif source == "primer":
                assert "path" in metadata
                assert "delimiter" in metadata
                assert "seed" in metadata
                assert "val_frac" in metadata

    def test_custom_delimiter(self, tmp_path):
        """Test that custom delimiter is passed through and used correctly."""
        # Create primer file with custom delimiter: 20 blocks, each 5k chars
        primer_file = tmp_path / "primer.txt"
        custom_delim = "---SPLIT---"
        big_dialogue = "dialogue content " + ("x" * 5000)
        blocks = [big_dialogue for _ in range(20)]
        primer_content = custom_delim.join(blocks)
        primer_file.write_text(primer_content)

        cache_dir = tmp_path / "cache"

        # Build with custom delimiter
        cfg = StreamBuildConfig(
            roam_root="nonexistent",
            primer_path=str(primer_file),
            cache_dir=str(cache_dir),
            delimiter=custom_delim,
            seed=42,
            required_sources=("primer",),
            enabled_sources=("primer",),
        )

        sources_train, sources_val = build_sources(cfg)

        # Verify primer was built
        assert "primer" in sources_train
        assert "primer" in sources_val

        # Verify metadata contains custom delimiter
        primer_meta_file = cache_dir / "primer_train.meta.json"
        assert primer_meta_file.exists()
        with open(primer_meta_file, "r") as f:
            meta = json.load(f)
        assert meta["delimiter"] == custom_delim

    def test_missing_required_source_fails(self, tmp_path):
        """Test that missing required source produces clear error."""
        cache_dir = tmp_path / "cache"

        # Config that requires roam but directory doesn't exist
        cfg = StreamBuildConfig(
            roam_root="nonexistent",
            primer_path="nonexistent",
            cache_dir=str(cache_dir),
            required_sources=("roam",),
            seed=42,
            enabled_sources=("roam",),
        )

        # Should fail with helpful error about roam
        with pytest.raises(RuntimeError) as exc_info:
            build_sources(cfg)

        error_msg = str(exc_info.value)
        assert "roam" in error_msg.lower()
        assert "required" in error_msg.lower() or "failed" in error_msg.lower()

    def test_required_roam_with_empty_val_raises(self, tmp_path):
        """Required roam should fail if val split would be empty."""
        cache_dir = tmp_path / "cache"
        roam_dir = tmp_path / "roam"
        roam_dir.mkdir()
        (roam_dir / "only.md").write_text("just one file")

        cfg = StreamBuildConfig(
            roam_root=str(roam_dir),
            primer_path="nonexistent",
            cache_dir=str(cache_dir),
            required_sources=("roam",),
            enabled_sources=("roam",),
            seed=42,
        )

        with pytest.raises(RuntimeError) as exc_info:
            build_sources(cfg)

        assert "validation" in str(exc_info.value).lower()

    def test_required_must_be_subset_of_enabled(self, tmp_path):
        """Test that required_sources outside enabled_sources raises early."""
        cache_dir = tmp_path / "cache"

        cfg = StreamBuildConfig(
            roam_root="nonexistent",
            primer_path="nonexistent",
            cache_dir=str(cache_dir),
            required_sources=("roam", "primer"),
            enabled_sources=("roam",),
        )

        with pytest.raises(ValueError) as exc_info:
            build_sources(cfg)

        assert "subset of enabled_sources" in str(exc_info.value)

    def test_allow_missing_sources(self, tmp_path):
        """Test that allow_missing_sources allows graceful degradation."""
        # Create only roam with multiple files to ensure both train and val splits
        roam_dir = tmp_path / "roam"
        roam_dir.mkdir()
        (roam_dir / "file1.md").write_text("roam content" * 500)
        (roam_dir / "file2.md").write_text("roam content" * 500)
        (roam_dir / "file3.md").write_text("roam content" * 500)

        cache_dir = tmp_path / "cache"

        cfg = StreamBuildConfig(
            roam_root=str(roam_dir),
            primer_path="nonexistent",
            cache_dir=str(cache_dir),
            allow_missing_sources=True,
            required_sources=(),
            seed=42,
            enabled_sources=("roam",),
        )

        sources_train, sources_val = build_sources(cfg)

        # Should have roam
        assert "roam" in sources_train
        # Primer should not be there (missing file)
        assert "primer" not in sources_train
        # Wiki disabled for this test, so it should not be present
