import random

from matplotlib import pyplot as plt
import numpy as np


def linearly_scaled_random(min_value, max_value, scaling_factor):
    """
    Generates a random number between min_value and max_value (inclusive),
    with a linearly scaled probability. The probability of the max_value is 
    'scaling_factor' times more than the min_value.

    Args:
    - min_value (int): The minimum value in the range.
    - max_value (int): The maximum value in the range.
    - scaling_factor (int): The factor by which the probability of the maximum
                            value is more likely than the minimum value.

    Returns:
    - int: A randomly selected number.
    """
    # # Example usage
    # min_num = 1
    # max_num = 100
    # scale_factor = 10
    # random_number = linearly_scaled_random(min_num, max_num, scale_factor)
    # print(f"Random number: {random_number}")

    if min_value >= max_value:
        raise ValueError("min_value must be less than max_value")
    if scaling_factor <= 0:
        raise ValueError("scaling_factor must be positive")

    # Create a range of values from min to max
    range_values = list(range(min_value, max_value + 1))

    # Create a linearly scaled weight list
    weights = [1 + (scaling_factor - 1) * (i - min_value) / (max_value - min_value) for i in range_values]

    # Select a value based on the weights
    return random.choices(range_values, weights=weights, k=1)[0]


def map_value_to_new_range(x, original_lower, original_upper, new_upper):
    """
    Maps a value from one range to another range.

    This function takes a value `x` that is within the range defined by 
    `original_lower` and `original_upper`, and maps it to a new range defined by 
    `original_lower` and `new_upper`.

    Parameters:
    x (float): The value to be mapped.
    original_lower (float): The lower bound of the original range.
    original_upper (float): The upper bound of the original range.
    new_upper (float): The upper bound of the new range.

    Returns:
    float: The value of `x` mapped to the new range.
    """

    # Example usage:
    #mapped_value = map_value_to_new_range(50, 0, 100, 8)
    #print("Mapped value:", mapped_value)

    return ((x - original_lower) / (original_upper - original_lower)) * (new_upper - original_lower) + original_lower

def gaussian_weighted_sampling(items, key, target_x, n, std_dev=10.0):
    """
    Samples n items from a list of dictionaries based on a Gaussian distribution centered around target_x.
    The function uses the value associated with the specified key in each dictionary for the Gaussian weighting.

    Args:
    items (list of dicts): List of dictionaries, each having a key that corresponds to a numeric value.
    key (str): The key to access the numeric value in the dictionaries.
    target_x (float): The target value to center the Gaussian distribution.
    n (int): The number of items to sample.
    std_dev (float): Standard deviation for the Gaussian distribution.

    Returns:
    list of dict: The selected items (dictionaries).
    """
    if not items or n > len(items):
        return None
    
    items = items.copy()  # Create a deep copy of items for each sampling

    sampled_items = []
    for _ in range(n):
        # Calculate Gaussian weights based on the value associated with the specified key
        weights = [np.exp(-((item[key] - target_x) ** 2) / (2 * std_dev ** 2)) for item in items]

        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Random weighted selection
        sampled_item = random.choices(items, weights=normalized_weights, k=1)[0]
        sampled_items.append(sampled_item)

        # Remove the sampled item from the list
        items.remove(sampled_item)

    return sampled_items

import matplotlib.pyplot as plt

def plot_gaussian_sampling_distribution(items, target_x, std_dev=6.0, num_samples=1000):
    """
    Plots the distribution of sampled items based on Gaussian distribution centered around target_x.

    Args:
    items (list): List of items, each having a `.cognitive_load` attribute.
    target_x (float): The target cognitive load to center the Gaussian distribution.
    std_dev (float): Standard deviation for the Gaussian distribution.
    num_samples (int): Number of samples to draw for the plot.
    """
    # Create a deep copy of items for each sampling
    sampled_items = [gaussian_weighted_sampling(items, key='cognitive_load', target_x=target_x, n=1, std_dev=std_dev)[0] for _ in range(num_samples)]

    # Count occurrences of each cognitive load
    cognitive_load_counts = {item['cognitive_load']: 0 for item in items}
    for item in sampled_items:
        cognitive_load_counts[item['cognitive_load']] += 1

    # Plotting
    plt.bar(cognitive_load_counts.keys(), cognitive_load_counts.values(), color='skyblue')
    plt.title(f'Distribution of Sampled Items (Target: {target_x}, Std Dev: {std_dev})')
    plt.xlabel('Cognitive Load')
    plt.ylabel('Frequency')
    plt.show()
    
def test_functions():
    # Define a list of dictionaries for testing
    items = [{'cognitive_load': i} for i in range(40)]

    # Define the parameters for the tests
    target_x = 20
    n = 10
    
    std_dev_values = [1.0, 3.0,4.0, 5.0, 6.0]
    num_samples = 1000

    # Test the functions with different std_dev values
    for std_dev in std_dev_values:
        print(f"Testing with std_dev = {std_dev}")

        # Test gaussian_weighted_sampling
        sampled_items = gaussian_weighted_sampling(items, key='cognitive_load', target_x=target_x, n=n, std_dev=std_dev)
        print(f"Sampled items: {sampled_items}")

        # Test plot_gaussian_sampling_distribution
        plot_gaussian_sampling_distribution(items, target_x=target_x, std_dev=std_dev, num_samples=num_samples)


#test_functions()