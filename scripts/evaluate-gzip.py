import logging
logging.basicConfig(level=logging.INFO)

from npc_gzip.compressors.base import BaseCompressor
from npc_gzip.compressors.gzip_compressor import GZipCompressor
from npc_gzip.knn_classifier import KnnClassifier

from src.datasets import CompressorKNNDataset
from src.metrics import compute_metrics
from src.callbacks import CustomNeptuneCallback
from src.plot_utils import plot_confusion_matrix
# from src.finetuners import Finetuner

import neptune
from neptune.types import File
import pandas as pd

# argparse
import argparse
# python train_sequence_classification_huggingface.py --model_name_or_path xlm-roberta-base \
# --num_labels 5 --train_dir data/train_th --eval_dir data/valid_th --num_train_epochs 3


# Optuna
import optuna
import neptune.integrations.optuna as optuna_utils

def main():
    parser = argparse.ArgumentParser(
        prog="evaluate-gzip.py",
        description="evaluate sequence classification with gzip-based kNN",
    )
    
    #required
    # parser.add_argument("--model_name_or_path", type=str,)
    # parser.add_argument("--num_labels", type=int,)
    parser.add_argument("--train_dir", type=str,)
    parser.add_argument("--valid_dir", type=str,)
    parser.add_argument("--test_dir", type=str,)


    #Dataset
    parser.add_argument("--csv_sep", type=str, default=',')
    parser.add_argument("--text_column_name", type=str,)
    parser.add_argument("--label_column_name", type=str,)


    #checkpoint
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument('--overwrite_output_dir', default=True, type=lambda x: (str(x).lower() in ['true', 't', 1, 'yes', 'y']))
    
    #tune hyperparameter with Optuna
    parser.add_argument("--optuna", default=False, type=lambda x: (str(x).lower() in ['true', 't', 1, 'yes', 'y']))
    parser.add_argument("--optuna_metric_for_best", type=str, default='f1')

    #hyperparameters
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--sampling_percentage", type=float, default=1.0)

    #Neptune AI
    parser.add_argument('--neptune_project', type=str)
    parser.add_argument('--neptune_api_token', type=str)
    parser.add_argument('--run_tags', nargs='*', default=[])
    
    args = parser.parse_args()

    neptune_project = args.neptune_project
    neptune_api_token = args.neptune_api_token
    run = neptune.init_run(
        project=neptune_project,
        api_token=neptune_api_token,
        tags=args.run_tags
    )
    neptune_callback = optuna_utils.NeptuneCallback(run)
    run['bash'].upload('./scripts/evaluate-gzip.sh')

    #datasets
    train_dataset = CompressorKNNDataset(
        args.train_dir,
        text_column_name=args.text_column_name,
        label_column_name=args.label_column_name,
        sep=args.csv_sep,
    )
    eval_dataset = CompressorKNNDataset(
        args.valid_dir,
        text_column_name=args.text_column_name,
        label_column_name=args.label_column_name,
        sep=args.csv_sep,
    )
    test_dataset = CompressorKNNDataset(
        args.test_dir,
        text_column_name=args.text_column_name,
        label_column_name=args.label_column_name,
        sep=args.csv_sep,
    )

    unique_labels = train_dataset.unique_labels

    compressor = GZipCompressor()
    gzip_model = KnnClassifier(
        compressor=compressor,
        training_inputs=train_dataset.texts,
        training_labels=train_dataset.labels,
        distance_metric='ncd'
    )

    top_k = args.top_k
    sampling_percentage = args.sampling_percentage


    print('args.optuna: ', args.optuna)

    if args.optuna:

        def objective(trial):
            
            top_k = trial.suggest_int('top_k', 1, 5)
            # sampling_percentage = trial.suggest_float('sampling_percentage', 0.8, 1.0)

            distances, pred, similar_samples = gzip_model.predict(
                eval_dataset.texts,
                top_k=top_k,
                sampling_percentage=args.sampling_percentage
            )

            metrics = compute_metrics(pred, eval_dataset.labels)
            score = metrics.get(args.optuna_metric_for_best, 0.0)

            return score
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=5, callbacks=[neptune_callback])

        # Get the best hyperparameters
        best_params = study.best_trial.params
        top_k = best_params['top_k']
        # sampling_percentage = best_params['sampling_percentage']

        print(f'The tuned hyperparameter is top_k: {top_k}')
        # print(f'The tuned hyperparameter is top_k: {top_k}, and sampling_percentage: {sampling_percentage}')


    print('top k: ', top_k, 'sampling percentage: ', sampling_percentage)

    distances, pred, similar_samples = gzip_model.predict(
        test_dataset.texts,
        top_k=top_k,
        sampling_percentage=sampling_percentage
    )

    metrics = compute_metrics(pred, test_dataset.labels)
    
    for metric_name, val in metrics.items():
        res = None
        if metric_name in ['f1', 'accuracy']:
            res = val
            run[f'gzip/eval/{metric_name}'] = res
        elif metric_name in ['confusion_matrix']:
            confusion_matrix_fig = plot_confusion_matrix(val, labels=unique_labels)   
            run[f'gzip/eval/{metric_name}'].upload(confusion_matrix_fig)
        elif metric_name in ['report']:
            report_df = pd.DataFrame(val).transpose()
            run[f'gzip/eval/{metric_name}'].upload(File.as_html(report_df))
    run.stop()

if __name__ == "__main__":
    main()