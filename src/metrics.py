import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from seqeval.metrics import (accuracy_score as seqeval_accuracy_score, 
                             classification_report as seqeval_classification_report, 
                             f1_score as seqeval_f1_score,
                             precision_score as seqeval_precision_score, 
                             recall_score as seqeval_recall_score)

def single_label_metrics(y_pred, y_labels, labels=None):
    print('y_labels: ', y_labels)
    print('y_pred: ', y_pred)
    f1_macro_average = f1_score(y_true=y_labels, y_pred=y_pred, average='macro')
    accuracy = accuracy_score(y_true=y_labels, y_pred=y_pred, normalize=True)
    conf = confusion_matrix(y_true=y_labels, y_pred=y_pred, labels=labels).tolist()
    report = classification_report(y_true=y_labels, y_pred=y_pred, output_dict=True)


    metrics = {'f1': f1_macro_average,
               'accuracy': accuracy,
               'confusion_matrix': conf,
               'report': report}
    return metrics

def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    y_pred = preds.argmax(-1)
    result = single_label_metrics(
        y_pred=y_pred, 
        y_labels=p.label_ids)
    return result

def compute_metrics(preds, labels):
    result = single_label_metrics(
        y_pred=preds,
        y_labels=labels)
    return result

def compute_metrics_with_labels(labels):
    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, 
                tuple) else p.predictions
        y_pred = preds.argmax(-1)
        result = single_label_metrics(
            y_pred=y_pred, 
            y_labels=p.label_ids,
            labels=list(labels))
        return result
    return compute_metrics