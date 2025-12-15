import pytest
from pathlib import Path
from niels_gpt.data import (
    load_wikitext,
    list_roam_paths,
    load_texts,
    split_roam_paths,
    load_primer_text,
    split_primer_dialogues,
    DIALOGUE_DELIM,
)


class TestWikitextLoader:
    """Test wikitext loader with monkeypatching (offline)"""

    def test_load_wikitext_keys(self, monkeypatch):
        """Ensure load_wikitext returns train/val/test keys"""

        class FakeDataset:
            def __init__(self):
                self.data = {
                    "train": {"text": ["line1", "line2"]},
                    "validation": {"text": ["val1", "val2"]},
                    "test": {"text": ["test1", "test2"]},
                }

            def __getitem__(self, key):
                return self.data[key]

        def mock_load_dataset(name, config):
            assert name == "wikitext"
            assert config == "wikitext-103-raw-v1"
            return FakeDataset()

        import niels_gpt.data.wikitext

        monkeypatch.setattr(niels_gpt.data.wikitext, "load_dataset", mock_load_dataset)

        result = load_wikitext()
        assert set(result.keys()) == {"train", "val", "test"}
        assert result["train"] == ["line1", "line2"]
        assert result["val"] == ["val1", "val2"]
        assert result["test"] == ["test1", "test2"]

    def test_load_wikitext_drops_empty_lines(self, monkeypatch):
        """Ensure empty-only strings are dropped"""

        class FakeDataset:
            def __init__(self):
                self.data = {
                    "train": {"text": ["line1", "   ", "", "line2", "\t\n"]},
                    "validation": {"text": ["val1", "  ", "val2"]},
                    "test": {"text": ["", "test1"]},
                }

            def __getitem__(self, key):
                return self.data[key]

        def mock_load_dataset(name, config):
            return FakeDataset()

        import niels_gpt.data.wikitext

        monkeypatch.setattr(niels_gpt.data.wikitext, "load_dataset", mock_load_dataset)

        result = load_wikitext()
        assert result["train"] == ["line1", "line2"]
        assert result["val"] == ["val1", "val2"]
        assert result["test"] == ["test1"]


class TestRoamFunctions:
    """Test roam markdown corpus functions"""

    def test_list_roam_paths(self, tmp_path):
        """Test that list_roam_paths returns sorted absolute paths for .md files only"""
        # Create directory structure
        (tmp_path / "a.md").write_text("a")
        (tmp_path / "b.md").write_text("b")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "c.md").write_text("c")
        (tmp_path / "non_md.txt").write_text("not markdown")

        paths = list_roam_paths(str(tmp_path))

        # Should only include .md files
        assert len(paths) == 3
        # All paths should be absolute
        assert all(Path(p).is_absolute() for p in paths)
        # Should be sorted
        basenames = [Path(p).name for p in paths]
        assert basenames == sorted(basenames)
        # Should not include non-md files
        assert not any("non_md.txt" in p for p in paths)

    def test_load_texts(self, tmp_path):
        """Test load_texts with utf-8 and invalid bytes"""
        # Create files with valid and invalid UTF-8
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"

        file1.write_text("Hello world", encoding="utf-8")
        # Write invalid UTF-8 bytes
        file2.write_bytes(b"Hello \xff\xfe invalid")

        paths = [str(file1), str(file2)]
        texts = load_texts(paths)

        assert len(texts) == 2
        assert texts[0] == "Hello world"
        # Should not crash due to errors="replace"
        assert "Hello" in texts[1]

    def test_split_roam_paths_deterministic(self, tmp_path):
        """Test that split is deterministic across calls"""
        # Create some test files
        for i in range(10):
            (tmp_path / f"file_{i}.md").write_text(f"content {i}")

        paths = list_roam_paths(str(tmp_path))

        train1, val1 = split_roam_paths(paths, seed=42)
        train2, val2 = split_roam_paths(paths, seed=42)

        assert train1 == train2
        assert val1 == val2

    def test_split_roam_paths_edge_cases(self, tmp_path):
        """Test split edge cases (< 2 files)"""
        # No files
        train, val = split_roam_paths([], seed=42)
        assert train == []
        assert val == []

        # One file
        (tmp_path / "single.md").write_text("single")
        paths = list_roam_paths(str(tmp_path))
        train, val = split_roam_paths(paths, seed=42)
        assert len(train) == 1
        assert len(val) == 0

    def test_split_roam_paths_disjoint(self, tmp_path):
        """Test that train and val are disjoint and cover all paths"""
        for i in range(20):
            (tmp_path / f"file_{i}.md").write_text(f"content {i}")

        paths = list_roam_paths(str(tmp_path))
        train, val = split_roam_paths(paths, seed=42, val_frac=0.2)

        # Disjoint
        assert set(train).isdisjoint(set(val))
        # Cover all paths
        assert set(train) | set(val) == set(paths)
        # Validation size is at least 1
        assert len(val) >= 1


class TestPrimerFunctions:
    """Test primer dialogue loading and splitting"""

    def test_load_primer_text(self, tmp_path):
        """Test load_primer_text reads file correctly"""
        primer_file = tmp_path / "test_primer.txt"
        content = "This is a test primer"
        primer_file.write_text(content, encoding="utf-8")

        result = load_primer_text(str(primer_file))
        assert result == content

    def test_split_primer_dialogues_drops_empty_blocks(self):
        """Test that empty blocks are dropped"""
        text = f"{DIALOGUE_DELIM}block1{DIALOGUE_DELIM}{DIALOGUE_DELIM}block2{DIALOGUE_DELIM}   {DIALOGUE_DELIM}block3{DIALOGUE_DELIM}"

        train, val = split_primer_dialogues(text, seed=42, val_frac=0.33)

        # Should have dropped empty blocks
        all_blocks = train.split(DIALOGUE_DELIM) + val.split(DIALOGUE_DELIM)
        for block in all_blocks:
            assert block.strip() != ""

        # Should have 3 non-empty blocks total
        train_blocks = [b for b in train.split(DIALOGUE_DELIM) if b.strip()]
        val_blocks = [b for b in val.split(DIALOGUE_DELIM) if b.strip()]
        assert len(train_blocks) + len(val_blocks) == 3

    def test_split_primer_dialogues_deterministic(self):
        """Test deterministic split across calls"""
        text = f"block1{DIALOGUE_DELIM}block2{DIALOGUE_DELIM}block3{DIALOGUE_DELIM}block4{DIALOGUE_DELIM}block5"

        train1, val1 = split_primer_dialogues(text, seed=42, val_frac=0.2)
        train2, val2 = split_primer_dialogues(text, seed=42, val_frac=0.2)

        assert train1 == train2
        assert val1 == val2

    def test_split_primer_dialogues_single_block(self):
        """Test that single block returns empty val"""
        text = "only one block"

        train, val = split_primer_dialogues(text, seed=42)

        assert train == "only one block"
        assert val == ""

    def test_split_primer_dialogues_no_leading_trailing_delim(self):
        """Test that output has no leading/trailing delimiter"""
        text = f"block1{DIALOGUE_DELIM}block2{DIALOGUE_DELIM}block3"

        train, val = split_primer_dialogues(text, seed=42, val_frac=0.33)

        # No leading/trailing delimiter
        assert not train.startswith(DIALOGUE_DELIM)
        assert not train.endswith(DIALOGUE_DELIM)
        assert not val.startswith(DIALOGUE_DELIM)
        assert not val.endswith(DIALOGUE_DELIM)

    def test_split_primer_dialogues_preserves_content(self):
        """Test that all content is preserved in split"""
        blocks = ["block1", "block2", "block3", "block4", "block5"]
        text = DIALOGUE_DELIM.join(blocks)

        train, val = split_primer_dialogues(text, seed=42, val_frac=0.2)

        train_blocks = set(train.split(DIALOGUE_DELIM))
        val_blocks = set(val.split(DIALOGUE_DELIM))

        # All blocks should be present
        assert train_blocks | val_blocks == set(blocks)
        # Disjoint
        assert train_blocks.isdisjoint(val_blocks)
