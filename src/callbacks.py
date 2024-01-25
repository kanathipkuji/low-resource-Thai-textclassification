from transformers import TrainerCallback
from transformers.integrations import NeptuneCallback
from neptune.types import File
import numpy as np
import pandas as pd

from src.plot_utils import plot_confusion_matrix

class CustomNeptuneCallback(NeptuneCallback):
    def __init__(self, *, run, api_token, project, labels):
        super().__init__(run=run, api_token=api_token, project=project)
        self.labels = labels
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if self._log_checkpoints == "best":
            best_metric_name = args.metric_for_best_model
            if not best_metric_name.startswith("eval_"):
                best_metric_name = f"eval_{best_metric_name}"

            metric_value = metrics.get(best_metric_name)

            operator = np.greater if args.greater_is_better else np.less

            self._should_upload_checkpoint = state.best_metric is None or operator(metric_value, state.best_metric)
        confusion_matrix = metrics['eval_confusion_matrix']
        confusion_matrix_fig = plot_confusion_matrix(confusion_matrix, self.labels)

        report = metrics['eval_report']
        report_df = pd.DataFrame(report).transpose()

        self.run['finetuning/eval/confusion_matrix'].upload(confusion_matrix_fig)
        self.run['finetuning/eval/report'].upload(File.as_html(report_df))