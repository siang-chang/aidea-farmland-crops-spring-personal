#!/usr/bin/env python
# coding: utf-8

from sklearn.metrics import classification_report

def evaluate(y_true, y_pred, zero_division = 1):
    labels = sorted(list(set(y_true + y_pred)))
    eval_result = classification_report(
        y_true, y_pred, target_names=labels, 
        output_dict=True, zero_division=zero_division
    )
    return eval_result

if __name__ == "__main__":
    # example 1
    y_true = [0, 1, 2, 0, 1, 2, 2, 2, 1]
    y_pred = [0, 2, 2, 0, 0, 1, 2, 0, 1]
    eval_result = evaluate(y_true, y_pred)
    
    # example 2
    y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
    y_pred = ["ant", "ant", "cat", "cat", "cat", "cat"]
    eval_result = evaluate(y_true, y_pred)
    