import random
import re

def preprocess_line(string):
    """
    Return a new string which removes all characters from the line that are not in the following set:
    characters in the English alphabet, space, digits, or the ‘.’ character.
    (That is, remove characters with accents and umlauts and the other punctuation marks).
    This function also lowercase all remaining characters and convert all digits to ‘0’.
    """
    string = string.lower()
    # Replace all digits with '0'
    string = re.sub(r'\d', '0', string)
    # Remove all characters that are not in the English alphabet, space, digits, or the '.' character
    string = re.sub(r'[^a-z0 .]', '', string)
    return string

def read_lines(filename):
    """
    Read the file and return lines of file (without '\n').
    """
    data = []
    with open(filename, mode='r') as f:
        for line in f:
            data.append(line.rstrip())
    return data

def divide_set(training_path, division_ratio=0.9, random_seed=42):
    """
    Divide the training dataset into training and validation sets based on division_ratio.
    Randomly shuffle the lines before splitting.

    Args:
        division_ratio (float): The ratio for splitting the dataset into training and validation sets.
        random_seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
            train_set: str
            val_set: str
    """
    random.seed(random_seed)
    lines = read_lines(training_path)
    random.shuffle(lines)

    train_len = int(len(lines) * division_ratio)
    train_set = lines[0:train_len]
    val_set = lines[train_len:]
    return train_set, val_set