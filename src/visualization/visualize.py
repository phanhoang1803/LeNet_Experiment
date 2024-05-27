# src/visualization/visualize.py
# Script for visualization (e.g., plotting training curves)

import matplotlib.pyplot as plt

def plot_training_curves(history, title):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()