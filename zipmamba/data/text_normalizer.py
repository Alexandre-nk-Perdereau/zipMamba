"""Text normalization for ASR training.

Normalizes text to better match what is actually spoken:
- Removes characters that aren't pronounced (middle dots, emojis, etc.)
- Normalizes punctuation (semicolons -> commas)
- Removes excessive whitespace
"""

import re
import unicodedata


# Characters to remove entirely (not pronounced)
REMOVE_CHARS = {
    "·",  # Middle dot (interpunct)
    "•",  # Bullet
    "◦",  # White bullet
    "‣",  # Triangular bullet
    "⁃",  # Hyphen bullet
    "※",  # Reference mark
    "†",  # Dagger
    "‡",  # Double dagger
    "§",  # Section sign
    "¶",  # Pilcrow
    "©",  # Copyright
    "®",  # Registered
    "™",  # Trademark
    "°",  # Degree (keep? depends on context)
    "|",  # Pipe
    "\\",  # Backslash
    "/",  # Forward slash (in isolation)
    "*",  # Asterisk
    "#",  # Hash
    "@",  # At sign
    "~",  # Tilde
    "^",  # Caret
    "_",  # Underscore
    "`",  # Backtick
    "=",  # Equals
    "+",  # Plus
    "<",  # Less than
    ">",  # Greater than
    "[",  # Open bracket
    "]",  # Close bracket
    "{",  # Open brace
    "}",  # Close brace
}

# Punctuation replacements (key -> value)
PUNCT_REPLACEMENTS = {
    ";": ",",
    "—": ",",  # Em dash -> comma
    "–": ",",  # En dash -> comma
    "«": '"',  # French quotes
    "»": '"',
    "‹": "'",
    "›": "'",
    '"': '"',  # Curly quotes -> straight
    '"': '"',
    """: "'",
    """: "'",
    "…": "...",  # Normalize ellipsis
}


def remove_emojis(text: str) -> str:
    """Remove emoji characters from text."""
    # Remove characters in emoji ranges
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # Emoticons
        "\U0001f300-\U0001f5ff"  # Symbols & pictographs
        "\U0001f680-\U0001f6ff"  # Transport & map
        "\U0001f1e0-\U0001f1ff"  # Flags
        "\U00002702-\U000027b0"  # Dingbats
        "\U000024c2-\U0001f251"  # Enclosed chars
        "\U0001f900-\U0001f9ff"  # Supplemental symbols
        "\U0001fa00-\U0001fa6f"  # Chess symbols
        "\U0001fa70-\U0001faff"  # Symbols extended
        "\U00002600-\U000026ff"  # Misc symbols
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def normalize_text(text: str) -> str:
    """Normalize text for ASR training.

    Args:
        text: Input text (potentially with special characters).

    Returns:
        Normalized text with unpronounced characters removed.
    """
    if not text:
        return text

    # Remove emojis
    text = remove_emojis(text)

    # Replace punctuation
    for old, new in PUNCT_REPLACEMENTS.items():
        text = text.replace(old, new)

    # Remove unpronounced characters
    text = "".join(c for c in text if c not in REMOVE_CHARS)

    # Normalize whitespace (multiple spaces -> single space)
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    # Remove leading/trailing punctuation that might be orphaned
    text = re.sub(r"^[,.\s]+", "", text)
    text = re.sub(r"[,.\s]+$", "", text)

    return text


def normalize_for_tokenization(text: str) -> str:
    """Additional normalization specifically for tokenization.

    More aggressive cleaning for cases where the tokenizer
    produces many tokens for special sequences.

    Args:
        text: Input text.

    Returns:
        Cleaned text optimized for tokenization.
    """
    text = normalize_text(text)

    # Collapse repeated punctuation (... -> .)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r",{2,}", ",", text)
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)

    # Remove isolated punctuation (punctuation with spaces on both sides)
    text = re.sub(r"\s+[.,!?]\s+", " ", text)

    # Final whitespace cleanup
    text = re.sub(r"\s+", " ", text).strip()

    return text
