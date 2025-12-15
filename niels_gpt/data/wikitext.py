from datasets import load_dataset


def load_wikitext() -> dict[str, list[str]]:
    """
    loads via datasets.load_dataset("wikitext", "wikitext-103-raw-v1")
    returns keys: "train", "val", "test"
    maps hf "validation" -> "val"
    drops empty-only lines: keep line iff line.strip() != ""
    """
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")

    result = {}

    # Process train split
    result["train"] = [line for line in ds["train"]["text"] if line.strip() != ""]

    # Process validation split -> "val"
    result["val"] = [line for line in ds["validation"]["text"] if line.strip() != ""]

    # Process test split
    result["test"] = [line for line in ds["test"]["text"] if line.strip() != ""]

    return result
