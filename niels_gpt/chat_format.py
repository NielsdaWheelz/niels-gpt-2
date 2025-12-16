def format_chat(messages: list[dict]) -> str:
    """
    messages: list of dicts with keys "role" and "content"
    returns transcript ending with exact prompt "assistant: " (no newline).
    """
    allowed_roles = {"system", "user", "assistant"}
    lines = []

    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]

        # Validate role
        if role not in allowed_roles:
            raise ValueError(f"Invalid role: {role}")

        # Check if this is the last message and it's an empty assistant message
        is_last = i == len(messages) - 1
        is_empty_assistant = role == "assistant" and content == ""

        if is_last and is_empty_assistant:
            # Special rule: treat as the prompt itself
            lines.append(f"{role}: ")
        else:
            # Normal message: include newline
            lines.append(f"{role}: {content}\n")

    # If the last message was NOT an empty assistant message, append the prompt
    if not (messages and messages[-1]["role"] == "assistant" and messages[-1]["content"] == ""):
        lines.append("assistant: ")

    return "".join(lines)


def extract_assistant_reply(generated_text: str) -> str:
    """
    generated_text includes the prompt prefix.
    find the last non-empty assistant reply, where a reply is the text after
    "assistant: " up to the earliest subsequent turn tag among:
      - "\nsystem: "
      - "\nuser: "
      - "\nassistant: "
    return value is stripped of leading/trailing whitespace via .strip().

    trailing empty "assistant: " prompts are ignored (common in generation).
    if "assistant: " is not present, raise ValueError.
    if all assistant occurrences are empty, print "(no reply)" and return "".
    """
    assistant_tag = "assistant: "
    turn_tags = ["\nsystem: ", "\nuser: ", "\nassistant: "]

    # Find all occurrences of "assistant: "
    occurrences = []
    idx = 0
    while True:
        pos = generated_text.find(assistant_tag, idx)
        if pos == -1:
            break
        occurrences.append(pos)
        idx = pos + 1

    if not occurrences:
        raise ValueError("assistant: not found in generated text")

    # For each occurrence, compute the candidate reply
    candidates = []
    for pos in occurrences:
        start = pos + len(assistant_tag)

        # Find the earliest next turn tag after this position
        earliest_tag_pos = len(generated_text)
        for tag in turn_tags:
            tag_pos = generated_text.find(tag, start)
            if tag_pos != -1 and tag_pos < earliest_tag_pos:
                earliest_tag_pos = tag_pos

        # Extract and strip the candidate
        candidate = generated_text[start:earliest_tag_pos].strip()
        candidates.append(candidate)

    # Return the last non-empty candidate, or "" if all are empty
    for candidate in reversed(candidates):
        if candidate:
            return candidate

    # If empty, print "(no reply)" and return ""
    print("(no reply)")
    return ""
