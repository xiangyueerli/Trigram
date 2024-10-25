import random

def maximum_likelihood_gen(trigram_probs):
    """
    Choose the next character based on the highest trigram probability.

    Args:
        trigram_probs (dict): Dictionary of {trigram : prob} where prob is the probability of the trigram.

    Returns:
        char: The next character, generated based on the two preceding characters.
    """
    # Get the corresponding key of the largest trigram_probs values
    chosen_trigram = max(trigram_probs, key=trigram_probs.get)
    # Return the last character of the chosen trigram
    return chosen_trigram[-1]


def weighted_random_gen(trigram_probs):
    """
    Randomly choose the next character based on the trigram probabilities using weighted random sampling.

    Args:
        trigram_probs (dict): Dictionary of {trigram : prob}.

    Returns:
        char: The next character, generated based on the two preceding characters.
    """
    # Extract characters and their probabilities
    trigrams = list(trigram_probs.keys())
    probabilities = list(float(prob) for prob in trigram_probs.values())

    # Randomly select a trigram based on the probabilities
    chosen_trigram = random.choices(trigrams, weights=probabilities, k=1)[0]

    return chosen_trigram[-1]


def top_k_gen(trigram_probs, k):
    """
    Get the trigram whose probability is in the top-k using the top-k sampling method.

    Args:
        trigram_probs (dict): Dictionary of {trigram : prob}.
        k (int): The number of top trigrams to consider (default is 8).

    Returns:
        char: The next character, generated based on the two preceding characters.
    """
    # Sort the tuple by tuple[1]:prob
    sorted_trigram = sorted(trigram_probs.items(), key=lambda item:item[1], reverse=True)
    # Use the top-k tuples
    sorted_trigram = sorted_trigram[:k]
    # Divide the tuple into trigram and probs
    trigram, probs = zip(*sorted_trigram)
    # Randomly select one element from the k sorted_trigram;
    chosen_trigram = random.choices(trigram, weights=probs, k=1)[0]
    return chosen_trigram[-1:]


def top_p_gen(trigram_probs, p):
    """
    Get the trigram whose probability is greater than or equal to p using the top-p sampling method.

    Args:
        trigram_probs (dict): Dictionary of {trigram : prob}.
        p (float): Probability threshold (default is 0.05).

    Returns:
        char: The next character, generated based on the two preceding characters.
    """
    higher_p_trigram = [item for item in trigram_probs.items() if item[1] >= p]
    # if none >= p: then turn to top_k_gen
    if len(higher_p_trigram) == 0:
        print(r'Warning: prop>=p not exists this time',end=' ')
        return top_k_gen(trigram_probs)

    # Divide the tuple into trigram and probs
    trigram, probs = zip(*higher_p_trigram)
    # Randomly select one element from the k sorted_trigram;
    chosen_trigram = random.choices(trigram, weights=probs, k=1)[0]
    return chosen_trigram[-1:]