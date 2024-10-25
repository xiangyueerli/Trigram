from collections import defaultdict

def good_turing_smoothing(trigram_counts):
    """
    Apply Good-Turing smoothing to adjust trigram counts.
    """
    Nc = defaultdict(int)
    bigram_counts = defaultdict(int)

    # Count the frequencies of frequencies (Nc)
    for count in trigram_counts.values():
        Nc[count] += 1

    # Apply Good-Turing smoothing to adjust counts
    adjusted_counts = {}
    for trigram, count in trigram_counts.items():
        if count + 1 in Nc and count in Nc:
            adjusted_count = (count + 1) * (Nc[count + 1] / Nc[count])
        else:
            adjusted_count = count  # If no information, keep the original count
        adjusted_counts[trigram] = adjusted_count
        bigram_counts[trigram[:2]] += adjusted_count

    # Update trigram counts with adjusted counts
    trigram_counts = adjusted_counts

    return trigram_counts, bigram_counts