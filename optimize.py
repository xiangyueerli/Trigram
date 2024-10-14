import numpy as np
from matplotlib import pyplot as plt
from model import Trigram


def optimize_a(start=0.01, end=1, interval=0.01, img_path="data/perplexity_optimization.png"):
    """
    This function optimizes the initial value (a) for a trigram model by calculating
    perplexity over a range of initial values and finding the minimum perplexity.

    Args:
        start (float): The starting value of 'a' to test.
        end (float): The ending value of 'a' to test.
        interval (float): The step size between each test value of 'a'.

    Returns:
        min_x (float): The value of 'a' that produces the minimum perplexity.
        min_y (float): The corresponding minimum perplexity value.
    """
    init_value_list = []
    perplexity_list = []

    for init_value in np.arange(start, end, interval):
        init_value_list.append(init_value)
        model = Trigram(init_value)
        # Train my own model: this is the standard model form.
        model.train()
        perplexity = model.validate()
        perplexity_list.append(perplexity)

    # Find the minimum value of y and its corresponding x value
    min_index = np.argmin(perplexity_list)
    min_x = init_value_list[min_index]
    min_y = perplexity_list[min_index]

    plt.plot(init_value_list, perplexity_list)

    # Mark the minimum value and location on the graph
    plt.scatter(min_x, min_y, color='red', label='Min value')
    plt.text(min_x, min_y, f'Min: ({min_x:.2f}, {min_y:.2f})',
             fontsize=10, ha='right', color='red')
    plt.legend()
    plt.title('Perplexity - a')

    plt.savefig(img_path)  # Save the figure as a .png file

    plt.show()
    return min_x, min_y
