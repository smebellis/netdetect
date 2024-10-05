import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, labels=None, filename='confusion_matrix.png'):
    """
    Plots and saves a confusion matrix using Seaborn's heatmap.

    :param y_true: List or NumPy array of actual labels.
    :param y_pred: List or NumPy array of predicted labels.
    :param labels: List of label names (optional).
    :param filename: Name of the file to save the plot (default: 'confusion_matrix.png').
    """
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # If labels aren't provided, create them based on unique values
    if labels is None:
        labels = np.unique(y_true)
    
    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    
    # Set plot labels
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # Save the plot as a file (e.g., png or jpg)
    plt.savefig(filename, dpi=300, bbox_inches='tight')  # dpi controls resolution, bbox_inches prevents clipping
    
    # Display the plot
    plt.show()


