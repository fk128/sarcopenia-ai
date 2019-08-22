import numpy as np

def dice(predictions, labelmap, labels=(0, 1)):
    """Calculates the categorical Dice similarity coefficients for each class
        between labelmap and predictions.
    Args:
        predictions (np.ndarray): predictions
        labelmap (np.ndarray): labelmap
        num_classes (int): number of classes to calculate the dice
            coefficient for
    Returns:
        np.ndarray: dice coefficient per class
    """
    num_classes = len(labels)
    dice_scores = np.zeros((num_classes))
    for i, l in enumerate(labels):
        tmp_den = (np.sum(predictions == l) + np.sum(labelmap == l))
        tmp_dice = 2. * np.sum((predictions == l) * (labelmap == l)) / \
                   tmp_den if tmp_den > 0 else 1.
        dice_scores[i] = tmp_dice
