from model import Trigram
from optimize import optimize_a

if __name__ == '__main__':
    # Add-a Smoothing
    # Get the init_value which make the perplexity smallest in the val_set
    # min_init_value, min_perplexity = optimize_a()

    # Use the best init_value: 'min_init_value' from the validation set.
    # model = Trigram(init_value=min_init_value, division_ratio=0.9)
    model = Trigram(init_value=0.1, division_ratio=0.9)
    model.train()
    model.generate()
    model.validate()
    model.test()

    # Good-Tuning Smoothing
    # model = Trigram(init_value=0, division_ratio=0.9)

