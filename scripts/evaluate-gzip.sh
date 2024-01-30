dataset_name='tscc'
shot='full'
text='filtered_fact'
label='label'
csv_sep=','
neptune_api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxZDIyNDdmMi05YTY3LTRmYTktODk3OC05ZmMxZmU1NmUxMjAifQ=='

python scripts/evaluate-gzip.py \
    --train_dir "./data/processed/$dataset_name/$dataset_name-$shot/train" \
    --valid_dir "./data/processed/$dataset_name/$dataset_name-$shot/valid" \
    --test_dir "./data/processed/$dataset_name/$dataset_name-$shot/test" \
    --text_column_name "$text" \
    --label_column_name "$label" \
    --csv_sep "$csv_sep" \
    --neptune_project kanathip137/tscc-gzip \
    --neptune_api_token "$neptune_api_token" \
    --run_tags gzip "$dataset_name" "$shot" \
    --sampling_percentage 1.0 \
    --top_k 1 \
    # --optuna T \