from typing import Iterator, Optional
import random
from datasets import load_dataset
from niels_gpt.special_tokens import assert_no_special_collision
from .types import ChatSample, ChatMessage


def _reconstruct_threads(
    records: list[dict],
    *,
    english_only: bool = True,
    max_messages: int = 32,
) -> list[dict]:
    """
    Reconstruct chat threads from oasst1 message tree structure.

    Sanitization:
      - Skips empty/whitespace-only messages
      - Skips threads with consecutive same-role messages (cannot safely merge)
      - Strips whitespace from message content

    Returns list of threads, where each thread is:
    {
        "messages": list[ChatMessage],
        "tree_id": str,
        "leaf_id": str,
        "message_ids": list[str]
    }
    """
    # Build lookups
    msg_by_id = {}
    children_by_parent = {}

    for record in records:
        msg_id = record["message_id"]

        # Filter by language if requested
        if english_only and record.get("lang") != "en":
            continue

        msg_by_id[msg_id] = record

        parent_id = record.get("parent_id")
        if parent_id is None:
            parent_id = "__root__"

        if parent_id not in children_by_parent:
            children_by_parent[parent_id] = []
        children_by_parent[parent_id].append(msg_id)

    # Find all root messages grouped by tree
    roots_by_tree = {}
    for msg_id, record in msg_by_id.items():
        if record.get("parent_id") is None:
            tree_id = record["message_tree_id"]
            if tree_id not in roots_by_tree:
                roots_by_tree[tree_id] = []
            roots_by_tree[tree_id].append(msg_id)

    # DFS to reconstruct all root->leaf paths
    threads = []

    def dfs(path: list[str], tree_id: str):
        """Recursively build paths from root to leaves."""
        if len(path) >= max_messages:
            # Hit max_messages limit, treat as leaf
            _emit_thread(path, tree_id)
            return

        current_id = path[-1]
        children = children_by_parent.get(current_id, [])

        if not children:
            # Reached a leaf
            _emit_thread(path, tree_id)
        else:
            # Continue down each child branch
            for child_id in children:
                dfs(path + [child_id], tree_id)

    def _emit_thread(path: list[str], tree_id: str):
        """Convert a message path into a ChatSample, with sanitization."""
        messages = []
        prev_role = None

        for msg_id in path:
            record = msg_by_id[msg_id]
            role = record["role"]
            text = record["text"]

            # Skip empty/whitespace-only messages
            if not text or not text.strip():
                continue

            assert_no_special_collision(
                text,
                dataset="oasst1",
                doc_index=msg_id,
                field=role,
            )

            # Map oasst roles to our role vocabulary
            if role == "prompter":
                mapped_role = "user"
            elif role == "assistant":
                mapped_role = "assistant"
            else:
                # Unexpected role, skip this thread entirely
                return

            # Check role alternation
            if prev_role is not None and mapped_role == prev_role:
                # Consecutive same-role messages - skip this thread entirely
                # (Cannot safely merge or skip individual messages without breaking context)
                return

            messages.append(ChatMessage(
                role=mapped_role,
                content=text.strip(),
            ))
            prev_role = mapped_role

        # Only emit thread if it has at least one message and passes all checks
        if messages:
            threads.append({
                "messages": messages,
                "tree_id": tree_id,
                "leaf_id": path[-1],
                "message_ids": path,
            })

    # Run DFS from each root
    for tree_id, root_ids in roots_by_tree.items():
        for root_id in root_ids:
            dfs([root_id], tree_id)

    return threads


def iter_oasst1_sft(
    *,
    split: str = "train",
    seed: int = 42,
    shuffle_trees: bool = True,
    take_trees: Optional[int] = None,
    english_only: bool = True,
    max_messages: int = 32,
) -> Iterator[ChatSample]:
    """
    loads OpenAssistant/oasst1 and reconstructs chat threads.

    reconstruction policy (locked):
      - build all root->leaf threads per message_tree_id (cap by max_messages)
      - if english_only: filter messages where lang == "en" (if field exists)
      - map roles:
          oasst role "prompter" -> "user"
          oasst role "assistant" -> "assistant"
      - do NOT invent system messages.

    meta must include: message_tree_id, leaf_message_id, and list of message_ids in the thread.
    """
    ds = load_dataset("OpenAssistant/oasst1", split=split)

    # Convert to list of records
    records = [dict(sample) for sample in ds]

    # Reconstruct threads
    threads = _reconstruct_threads(
        records,
        english_only=english_only,
        max_messages=max_messages,
    )

    # Shuffle at tree level if requested
    if shuffle_trees:
        rng = random.Random(seed)
        rng.shuffle(threads)

    # Yield threads
    count = 0
    for thread in threads:
        if take_trees is not None and count >= take_trees:
            break

        yield ChatSample(
            messages=thread["messages"],
            source="oasst1",
            meta={
                "message_tree_id": thread["tree_id"],
                "leaf_message_id": thread["leaf_id"],
                "message_ids": thread["message_ids"],
            },
        )
        count += 1
