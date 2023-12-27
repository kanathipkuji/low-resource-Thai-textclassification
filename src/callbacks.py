from transformers import TrainerCallback
from neptune.types import File
import numpy as np
import pandas as pd

from .plot_utils import plot_confusion_matrix

class CustomNeptuneCallback(TrainerCallback):
    def __init__(self, run, labels):
        super().__init__()
        self.run = run
        self.labels = labels
    def on_evaluate(self, args, states, control, metrics, **kwargs):
        confusion_matrix = metrics['eval_confusion_matrix']
        confusion_matrix_fig = plot_confusion_matrix(confusion_matrix, self.labels)

        report = metrics['eval_report']
        report_df = pd.DataFrame(report).transpose()

        self.run['finetuning/eval/confusion_matrix'].upload(confusion_matrix_fig)
        self.run['finetuning/eval/report'].upload(File.as_html(report_df))