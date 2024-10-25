import random
from collections import defaultdict
import math

import sampling
import smoothing
from utils import read_lines, preprocess_line, divide_set


class Trigram:
    """
    Trigram language model for character-level language modeling.

    Attributes:
        init_value (int): Initial value to be added to all bigram and trigram counts if using add-a smoothing.
        division_ratio (float): The ratio for splitting the dataset into training and validation sets.
        model_my_path (str): Path to the model file for 'my' trigram model.
        training_path (str): Path to the training dataset.
        test_path (str): Path to the test dataset.
        model_br_path (str): Path to the 'br' trigram model file.
        smoothing (str): If using smoothing, specify the smoothing parameter.
        random_seed (int, optional): Random seed for reproducibility. Default is 42.
    """
    def __init__(self, init_value=0, division_ratio=0.9, model_my_path = 'model/model-en.en', training_path ='data/training.en',
                 test_path = 'data/test', model_br_path = 'model/model-br.en', smoothing='', random_seed=42):
        """
        Initialize the Trigram model with specified parameters.

        Args:
            init_value (int): Initial value to be added to all bigram and trigram counts if using add-a smoothing.
            division_ratio (float): The ratio for splitting the dataset into training and validation sets.
            model_my_path (str): Path to the model file for 'my' trigram model.
            training_path (str): Path to the training dataset.
            test_path (str): Path to the test dataset.
            model_br_path (str): Path to the 'br' trigram model file.
            smoothing (str): If using smoothing, specify the smoothing parameter.
            random_seed (int, optional): Random seed for reproducibility. Default is 42.
        """
        self.init_value = init_value
        self.division_ratio = division_ratio
        self.model_my_path = model_my_path
        self.training_path = training_path
        self.test_path = test_path
        self.model_br_path = model_br_path
        self.smoothing = smoothing
        self.random_seed = random_seed

        # Set random seed for reproducibility
        random.seed(self.random_seed)

        if division_ratio >= 1 or division_ratio <= 0:
            raise ValueError(f"Parameter 'division_ratio' must be set in (0, 1)")

        self.bigram_counts = defaultdict(int)
        self.trigram_counts = defaultdict(int)
        self.trigram_model = defaultdict(float)
        self.voca = [' ', '#', '.', '0', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                     'q',
                     'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        self.train_set, self.val_set = divide_set(self.training_path, self.division_ratio)
        self.test_set = read_lines(self.test_path)

        self.get_trigram_counts()

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
            for trigram, prob in self.trigram_model.items():
                line = f'{trigram} {prob:.2e}\n'
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

    def train(self):
        """
        Train the trigram model by calculating probabilities for each trigram.
        """
        print('Training...')

        # Apply Good-Turing smoothing before training
        if self.smoothing == 'Good Turing':
            print(1)
            self.trigram_counts, self.bigram_counts = smoothing.good_turing_smoothing(self.trigram_counts)

        # Convert counts to probabilities and normalize
        for trigram, trigram_num in self.trigram_counts.items():
            bigram = trigram[:2]
            bigram_num = self.bigram_counts[bigram]

            self.trigram_model[trigram] = trigram_num / bigram_num

        self.write_model()
        print('Training completed!')
        print()

    def compute_perplexity_term_per_line(self, test_sentence):
        """
        Compute the perplexity term for a single test sentence.

        Args:
            test_sentence (str): The sentence to compute perplexity for.

        Returns:
            float: The perplexity of the test sentence.
        """
        # Preprocess the test sentence
        test_sentence = preprocess_line(test_sentence)
        test_sentence = f'##{test_sentence}#'

        perplexity_term = 0.0
        trigram_num = len(test_sentence) - 2

        for i in range(len(test_sentence) - 2):
            trigram = test_sentence[i:i + 3]
            perplexity_term -= math.log2(self.trigram_model[trigram])

        return perplexity_term, trigram_num

    def compute_perplexity(self, model_path, is_val=False):
        """
        Compute the perplexity for the validation set or test set.

        Args:
            model_path (str): The path to the model file to load.
            is_val (bool): Whether to compute perplexity on the validation set.

        Returns:
            float: The average perplexity of the sentences in the set.
        """
        if is_val:
            test_set = self.val_set
            print(f'in validation set, ', end='')
        else:
            test_set = self.test_set
            print(f'in test set, ', end='')
        self.read_model(model_path)

        sum_perplexity_term = 0
        sum_trigram_num = 0
        for sentence in test_set:
            perplexity_term, trigram_num = self.compute_perplexity_term_per_line(sentence)
            sum_trigram_num += trigram_num
            sum_perplexity_term += perplexity_term
        perplexity = 2 ** (sum_perplexity_term / sum_trigram_num)
        print(f'perplexity: {perplexity}')
        return perplexity

    def validate(self):
        """
        Validate the model on the validation set.

        Returns:
            float: The average perplexity on the validation set.
        """
        print(f'For {self.model_my_path}, ', end='')
        return self.compute_perplexity(self.model_my_path, is_val=True)

    def test(self):
        """
        Test the model on the test set and print the results for model-br and model-my.
        """
        print(f'For {self.model_br_path}, ', end='')
        self.compute_perplexity(self.model_br_path)
        # Test model-br.en
        print(f'For {self.model_my_path}, ', end='')
        self.compute_perplexity(self.model_my_path)
        print()

    def generate_from_lm(self, model_path, length=300, sampling_method_num=0, k=8, p=0.05):
        """
        Generate a sequence of characters from the language model.

        Args:
            model_path (str): Path to the model file.
            length (int): Length of the generated sequence (default 300).
            sampling_method_num (int): Sampling method to use (default 0).
            k (int): The k value for top-k sampling (default 8).
            p (float): The p value for top-p sampling (default 0.05).
        """
        # Read the trigram model from the file
        self.read_model(model_path)

        # Start the generated text with "##" as it's common to start with boundary symbols
        current_bigram = "##"
        generated_text = current_bigram

        i = 0
        generate_text_list = []
        # Start with two characters so generate (length - 2) more
        while True:
            # Get the dict whose elements are {trigram: prob} and trigram[:2]==current_bigram
            trigram_probs = {trigram: prob for trigram, prob in self.trigram_model.items() if trigram[:2] == current_bigram}

            if i == length:
                if generated_text[-1] == '#':
                    generate_text_list.append(generated_text[2:-1])
                else:
                    generate_text_list.append(generated_text[2:])
                break

            # If no matching trigram found, stop early
            # Not exist: 1.Ending'_#_'(Except for beginning '##_'); 2.Only one char in a sentence'#_#'.
            if not trigram_probs:
                generate_text_list.append(generated_text[2:-1])
                current_bigram = "##"
                generated_text = current_bigram
                continue

            next_char = ''
            # 4 sampling methods
            if sampling_method_num == 0:
                next_char = sampling.weighted_random_gen(trigram_probs)
            if sampling_method_num == 1:
                next_char = sampling.top_k_gen(trigram_probs, k)
            if sampling_method_num == 2:
                next_char = sampling.top_p_gen(trigram_probs, p)
            if sampling_method_num == 3:
                next_char = sampling.maximum_likelihood_gen(trigram_probs)

            # Append the next character to the generated text
            generated_text += next_char
            if next_char != '#':
                i += 1

            # Update the current bigram to the last two characters of the generated text
            current_bigram = generated_text[-2:]

        print(f'{len(generate_text_list)} sentence generated:')
        for generated_text in generate_text_list:
            print(generated_text)

    def generate(self, sampling_method_num=0, k=10, p=0.05):
        """
        Generate text with model-br and model-my.

        Args:
            sampling_method_num (int): Sampling method to use (default 0).
            k (int): The k value for top-k sampling (default 8).
            p (float): The p value for top-p sampling (default 0.05).

        """
        print(f'-----Generating text:--------\nFrom {self.model_br_path}: ')
        self.generate_from_lm(self.model_br_path, sampling_method_num=sampling_method_num, k=k, p=p)
        print(f'From {self.model_my_path}: ')
        self.generate_from_lm(self.model_my_path, sampling_method_num=sampling_method_num, k=k, p=p)
        print()
