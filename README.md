# Trigram Language Model Project

## Overview
This project implements a trigram language model for character-level language modeling. Models are trained respectively on English, German, and Spanish datasets using various smoothing techniques and sampling methods. The project includes methods for model training, text generation, perplexity calculation, and model optimization.

## File Structure

1. **model.py**: Defines the `Trigram` class, handling model initialization, training, text generation, and perplexity calculation.
2. **main.py**: Runs the complete pipeline, including model training, text generation with various sampling methods, testing, and optimization.
3. **optimize.py**: Provides the `optimize_a` function to find the best smoothing parameter `a` that minimizes perplexity.
4. **smoothing.py**: Implements Good-Turing smoothing to adjust trigram counts for unseen events.
5. **sampling.py**: Offers different text sampling methods, including maximum likelihood, top-k, top-p, and weighted random generation.
6. **utils.py**: Contains helper functions for preprocessing text, reading data files, and splitting datasets for training and validation.

## Requirements

- **Python 3.9** (Recommended to run in PyCharm)

## Folder Setup
To get started, ensure that the following folder structure is in place:

1. **data/**: Place the training datasets (`training.de`, `training.en`, `training.es`) in this folder.
2. **model/**: Place the initial model file `model-br.en` here. Trained models will also be saved in this folder.
3. **output/**: This folder should be empty initially. Generated output files, such as perplexity optimization plots, will be saved here.

### Running Extra Questions
If you need to run the extra question (Q6), make sure the `data/test-port` file is included in the `data/` folder. The content for this file is provided at the end of the Appendix.

## How to Run

1. Copy the provided code into the appropriate files (`main.py`, `model.py`, `optimize.py`, etc.).
2. Ensure the `data/`, `model/`, and `output/` folders are correctly set up as described above.
3. To run the project and get the results for Q1-Q5, simply run the `main.py` file. This will train the models, perform optimization, and output the necessary results.
   ```bash
   python main.py
   ```
4. To run the extra question, Good-Turing smoothing, or other sampling methods, uncomment the relevant code in `main.py` before running.

## Appendix

### `data/test-port` content:
```
(Include the content of `test-port` here if applicable)
```
