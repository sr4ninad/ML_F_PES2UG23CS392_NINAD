# student_lab.py
import torch

def get_entropy_of_dataset(tensor: torch.Tensor):
    """
    Calculate the entropy of the entire dataset.
    Formula: Entropy = -Σ(p_i * log2(p_i)) where p_i is the probability of class i
    """
    # Get the target column (last column)
    target = tensor[:, -1]

    # Count unique classes
    classes, counts = torch.unique(target, return_counts=True)
    probabilities = counts.float() / target.shape[0]

    # Entropy = -sum(p * log2(p)), skip zero probs
    entropy = -torch.sum(probabilities * torch.log2(probabilities))

    return float(entropy)


def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int):
    """
    Calculate the average information (weighted entropy) of an attribute.
    Formula: Avg_Info = Σ((|S_v|/|S|) * Entropy(S_v)) where S_v is subset with attribute value v.
    """
    total_rows = tensor.shape[0]
    attribute_values = tensor[:, attribute]
    unique_vals, counts = torch.unique(attribute_values, return_counts=True)

    avg_info = 0.0
    for val, count in zip(unique_vals, counts):
        subset = tensor[attribute_values == val]
        subset_entropy = get_entropy_of_dataset(subset)
        weight = count.item() / total_rows
        avg_info += weight * subset_entropy

    return float(avg_info)


def get_information_gain(tensor: torch.Tensor, attribute: int):
    """
    Calculate Information Gain for an attribute.
    Formula: Information_Gain = Entropy(S) - Avg_Info(attribute)
    """
    total_entropy = get_entropy_of_dataset(tensor)
    avg_info = get_avg_info_of_attribute(tensor, attribute)
    info_gain = total_entropy - avg_info
    return round(float(info_gain), 4)


def get_selected_attribute(tensor: torch.Tensor):
    """
    Select the best attribute based on highest information gain.

    Returns:
        tuple: (dict of attribute:index -> information gain, index of best attribute)
    """
    num_attributes = tensor.shape[1] - 1  # last column is target
    info_gains = {}

    for attr in range(num_attributes):
        ig = get_information_gain(tensor, attr)
        info_gains[attr] = ig

    # Attribute with max info gain
    best_attr = max(info_gains, key=info_gains.get)

    return info_gains, best_attr
