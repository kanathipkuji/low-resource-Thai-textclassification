dataset_name='tscc'
shot='5'
text='filtered_fact'
label='label'
csv_sep=','
neptune_api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxZDIyNDdmMi05YTY3LTRmYTktODk3OC05ZmMxZmU1NmUxMjAifQ=='

python scripts/train-finetuner.py \
    --train_dir "./data/processed/$dataset_name/$dataset_name-$shot/train" \
    --valid_dir "./data/processed/$dataset_name/$dataset_name-$shot/valid" \
    --test_dir "./data/processed/$dataset_name/$dataset_name-$shot/test" \
    --text_column_name $text \
    --label_column_name $label \
    --csv_sep "$csv_sep" \
    --deterministic F \
    --output_dir "./results/$dataset_name-$shot" \
    --logging_dir "./results/$dataset_name-$shot-logs" \
    --logging_steps 5 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --num_train_epochs 50 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --warmup_steps 500 \
    --weight_decay 0.01 \
    --neptune_project kanathip137/$dataset_name-finetuner \
    --neptune_api_token $neptune_api_token \
    --run_tag "$shot-shot"
    # --ib True \
    