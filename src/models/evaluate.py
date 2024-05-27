# src/models/evaluate.py
# Script for evaluating the model

import tensorflow as tf

def evaluate_model(model, test_dataset):
    loss, accuracy = model.evaluate(test_dataset)
    
    return loss, accuracy