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
            data.append(line.strip())
    return data
