"""ASR evaluation metrics: WER and CER."""

from typing import Optional


def _levenshtein_distance(ref: list, hyp: list) -> int:
    """Compute Levenshtein distance between two sequences.

    Args:
        ref: Reference sequence.
        hyp: Hypothesis sequence.

    Returns:
        Edit distance.
    """
    m, n = len(ref), len(hyp)

    # DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],  # Deletion
                    dp[i][j - 1],  # Insertion
                    dp[i - 1][j - 1],  # Substitution
                )

    return dp[m][n]


def compute_wer(
    references: list[str],
    hypotheses: list[str],
    return_details: bool = False,
) -> float | dict:
    """Compute Word Error Rate (WER).

    WER = (S + D + I) / N
    where S = substitutions, D = deletions, I = insertions, N = reference words

    Args:
        references: List of reference transcripts.
        hypotheses: List of hypothesis transcripts.
        return_details: If True, return dict with detailed stats.

    Returns:
        WER as float (0-1), or dict with details if return_details=True.
    """
    if len(references) != len(hypotheses):
        raise ValueError("Number of references and hypotheses must match")

    total_errors = 0
    total_words = 0

    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.strip().lower().split()
        hyp_words = hyp.strip().lower().split()

        errors = _levenshtein_distance(ref_words, hyp_words)
        total_errors += errors
        total_words += len(ref_words)

    wer = total_errors / max(1, total_words)

    if return_details:
        return {
            "wer": wer,
            "errors": total_errors,
            "words": total_words,
        }

    return wer


def compute_cer(
    references: list[str],
    hypotheses: list[str],
    return_details: bool = False,
) -> float | dict:
    """Compute Character Error Rate (CER).

    CER = (S + D + I) / N
    where S = substitutions, D = deletions, I = insertions, N = reference chars

    Args:
        references: List of reference transcripts.
        hypotheses: List of hypothesis transcripts.
        return_details: If True, return dict with detailed stats.

    Returns:
        CER as float (0-1), or dict with details if return_details=True.
    """
    if len(references) != len(hypotheses):
        raise ValueError("Number of references and hypotheses must match")

    total_errors = 0
    total_chars = 0

    for ref, hyp in zip(references, hypotheses):
        ref_chars = list(ref.strip().lower())
        hyp_chars = list(hyp.strip().lower())

        errors = _levenshtein_distance(ref_chars, hyp_chars)
        total_errors += errors
        total_chars += len(ref_chars)

    cer = total_errors / max(1, total_chars)

    if return_details:
        return {
            "cer": cer,
            "errors": total_errors,
            "chars": total_chars,
        }

    return cer


def compute_metrics(
    references: list[str],
    hypotheses: list[str],
) -> dict[str, float]:
    """Compute all ASR metrics.

    Args:
        references: List of reference transcripts.
        hypotheses: List of hypothesis transcripts.

    Returns:
        Dict with WER and CER.
    """
    return {
        "wer": compute_wer(references, hypotheses),
        "cer": compute_cer(references, hypotheses),
    }
