from collections import defaultdict
import math
import sampling
from utils import read_lines, preprocess_line


class Trigram:
    def __init__(self, init_value, division_ratio=0.9, model_my_path = 'model/model-my.en', training_en_path = 'data/training.en',
                 test_path = 'data/test', model_br_path = 'model/model-br.en'):
        """
        Initialize the Trigram model with specified parameters.

        Args:
            init_value (int): Initial value to be added to all bigram and trigram counts.
            division_ratio (float): The ratio for splitting the dataset into training and validation sets.
            model_my_path (str): Path to the model file for 'my' trigram model.
            training_en_path (str): Path to the training dataset.
            test_path (str): Path to the test dataset.
            model_br_path (str): Path to the 'br' trigram model file.
        """
        self.init_value = init_value
        self.division_ratio = division_ratio
        self.model_my_path = model_my_path
        self.training_en_path = training_en_path
        self.test_path = test_path
        self.model_br_path = model_br_path

        if division_ratio >= 1 or division_ratio <= 0:
            raise ValueError(f"Parameter 'division_ratio' must be set in (0, 1)")

        self.bigram_counts = defaultdict(int)
        self.trigram_counts = defaultdict(int)
        self.trigram_model = defaultdict(int)
        self.voca = [' ', '#', '.', '0', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                     'q',
                     'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        self.train_set = []
        self.val_set = []
        self.test_set = []

        self.divide_set()
        self.get_trigram_counts()

    def divide_set(self):
        """
        Divide the training dataset into training and validation sets based on division_ratio.
        """
        lines = read_lines(self.training_en_path)

        train_len = int(len(lines) * self.division_ratio)
        self.train_set = lines[0:train_len]
        self.val_set = lines[train_len:]

        self.test_set = read_lines(self.test_path)

    def read_model(self, filename):
        """
        Read a trigram model from a file.

        Args:
            filename (str): The path to the model file.
        """
        with open(filename, mode='r') as f:
            for line in f:
                trigram = line[0:3]
                prop = float(line[4:].strip())
                self.trigram_model[trigram] = prop

    def write_model(self):
        """
        Write the trigram model to a file (self.model_my_path).
        """
        save_path = self.model_my_path
        with open(save_path, 'w', encoding='utf-8') as f:
            for trigram, count in self.trigram_model.items():
                line = f'{trigram} {count}\n'
                f.write(line)

    def init_dict(self):
        """
        Initialize bigram and trigram counts with the initial value for smoothing.
        """
        for c1 in self.voca:
            for c2 in self.voca:
                for c3 in self.voca:
                    if c2 == '#' and c1 != '#' or c1 == '#' and c3 == '#':
                        continue
                    bigram = c1 + c2
                    trigram = bigram + c3

                    self.bigram_counts[bigram] += self.init_value
                    self.trigram_counts[trigram] += self.init_value

    def get_trigram_counts(self):
        """
        Count the occurrences of bigrams and trigrams in the training data.
        """
        self.init_dict()

        for line in self.train_set:
            # Preprocess the line and add '#' at the start and end
            line = preprocess_line(line)
            line = f'##{line}#'
            # Get the counts of trigram
            for i in range(len(line) - 2):
                bigram = line[i:i + 2]
                trigram = line[i:i + 3]
                self.bigram_counts[bigram] += 1
                self.trigram_counts[trigram] += 1

    def good_turing_smoothing(self):
        """
        Apply Good-Turing smoothing to adjust trigram counts.
        """
        Nc = defaultdict(int)
        # Count the frequencies of frequencies (Nc)
        for count in self.trigram_counts.values():
            Nc[count] += 1

        # Apply Good-Turing smoothing to adjust counts
        adjusted_counts = {}
        for trigram, count in self.trigram_counts.items():
            if count + 1 in Nc and count in Nc:
                adjusted_count = (count + 1) * (Nc[count + 1] / Nc[count])
            else:
                adjusted_count = count  # If no information, keep the original count
            adjusted_counts[trigram] = adjusted_count

        # Update trigram counts with adjusted counts
        self.trigram_counts = adjusted_counts

    def train(self):
        """
        Train the trigram model by calculating probabilities for each trigram.
        """
        print('Training...')

        # Apply Good-Turing smoothing before training
        # self.good_turing_smoothing()

        # Convert counts to probabilities and normalize
        for trigram, trigram_num in self.trigram_counts.items():
            bigram = trigram[:2]
            bigram_num = self.bigram_counts[bigram]

            self.trigram_model[trigram] = f"{trigram_num / bigram_num:.2e}"

        self.write_model()
        print('Training completed!')
        print()

    def compute_perplexity_per_line(self, test_sentence):
        """
        Compute perplexity for a single test sentence.

        Args:
            test_sentence (str): The sentence to compute perplexity for.

        Returns:
            float: The perplexity of the test sentence.
        """
        # Preprocess the test sentence
        test_sentence = preprocess_line(test_sentence)
        test_sentence = f'##{test_sentence}#'

        perplexity = 0.0
        N = len(test_sentence) - 2

        for i in range(len(test_sentence) - 2):
            trigram = test_sentence[i:i + 3]
            # print(self.trigram_model[trigram])
            perplexity += -math.log2(self.trigram_model[trigram])

        perplexity = 2 ** (perplexity / N)
        return perplexity

    def compute_perplexity(self, model_path, is_val=False):
        """
        Compute the average perplexity for the validation or test set.

        Args:
            model_path (str): The path to the model file to load.
            is_val (bool): Whether to compute perplexity on the validation set.

        Returns:
            float: The average perplexity of the sentences in the set.
        """
        if is_val == True:
            test_set = self.val_set
            print(f'in validation set, ', end='')
        else:
            test_set = self.test_set
            print(f'in test set, ', end='')
        self.read_model(model_path)

        sum_perplexity = 0
        for idx, sentence in enumerate(test_set):
            perplexity = self.compute_perplexity_per_line(sentence)
            sum_perplexity = sum_perplexity + perplexity
        avg_perplexity = sum_perplexity / len(test_set)
        print(f'avg perplexity: {avg_perplexity}')
        return avg_perplexity

    def validate(self):
        """
        Validate the model on the validation set.

        Returns:
            float: The average perplexity on the validation set.
        """
        print('For model-my, ', end='')
        return self.compute_perplexity(self.model_my_path, is_val=True)

    def test(self):
        """
        Test the model on the test set and print the results for model-br and model-my.
        """
        print('For model-br, ', end='')
        self.compute_perplexity(self.model_br_path)
        # Test model-br.en
        print('For model-my, ', end='')
        self.compute_perplexity(self.model_my_path)
        print()

    def generate_from_lm(self, model_path, length=300, sampling_method_num=0):
        """
        Generate a sequence of characters from the language model.

        Args:
            model_path (str): Path to the model file.
            length (int): Length of the generated sequence (default 300).
            sampling_method_num (int): Sampling method to use (default 0).
        """
        # Read the trigram model from the file
        self.read_model(model_path)

        # Start the generated text with "##" as it's common to start with boundary symbols
        current_bigram = "##"
        generated_text = current_bigram

        # Start with two characters so generate (length - 2) more
        for i in range(length - 2):
            # Get the dict whose elements are {trigram: prob} and trigram[:2]==current_bigram
            trigram_probs = {trigram: prob for trigram, prob in self.trigram_model.items() if trigram[:2] == current_bigram}

            # If no matching trigram found, stop early
            # Not exist: 1.Ending'_#_'(Except for beginning '##_'); 2.Only one char in a sentence'#_#'
            if not trigram_probs:
                break

            # 4 sampling methods
            if sampling_method_num == 0:
                next_char = sampling.weighted_random_gen(trigram_probs)
            if sampling_method_num == 1:
                next_char = sampling.top_k_gen(trigram_probs, k=10)
            if sampling_method_num == 2:
                next_char = sampling.top_p_gen(trigram_probs, p=0.05)
            if sampling_method_num == 3:
                next_char = sampling.maximum_likelihood_gen(trigram_probs)

            # Append the next character to the generated text
            generated_text += next_char

            # Update the current bigram to the last two characters of the generated text
            current_bigram = generated_text[-2:]

        print(generated_text)

    def generate(self):
        """
        Generate text with model-br and model-my.
        """
        print('Generating text: \nFrom model-br: ', end='')
        self.generate_from_lm(self.model_br_path)
        print('From model-my: ', end='')
        self.generate_from_lm(self.model_my_path)
        print()
