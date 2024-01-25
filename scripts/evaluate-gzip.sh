dataset_name='wongnai'
shot='5'
text='review'
label='star'
csv_sep=';'
neptune_api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxZDIyNDdmMi05YTY3LTRmYTktODk3OC05ZmMxZmU1NmUxMjAifQ=='

python scripts/evaluate-gzip.py \
    --train_dir "./data/processed/$dataset_name/$dataset_name-$shot/train" \
    --valid_dir "./data/processed/$dataset_name/$dataset_name-$shot/valid" \
    --test_dir "./data/processed/$dataset_name/$dataset_name-$shot/test" \
    --text_column_name "$text" \
    --label_column_name "$label" \
    --csv_sep "$csv_sep" \
    --neptune_project kanathip137/$dataset_name-$shot-gzip \
    --neptune_api_token "$neptune_api_token" \
    --sampling_percentage 1.0 \
    --top_k 1 \
    # --optuna T \