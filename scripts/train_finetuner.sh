python scripts/train_finetuner.py \
    --train_dir './data/processed/tscc/train' \
    --valid_dir './data/processed/tscc/valid' \
    --test_dir './data/processed/tscc/test' \
    --text_column_name filtered_fact \
    --label_column_name label \
    --output_dir './results/finetuned-rel-art-pred' \
    --logging_dir './results/finetuned-rel-art-pred-logs' \
    --logging_steps 5 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --num_train_epochs 50 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --warmup_steps 500 \
    --weight_decay 0.01 \
    --neptune_project kanathip137/finetuned-tscc \
    --neptune_api_token eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxZDIyNDdmMi05YTY3LTRmYTktODk3OC05ZmMxZmU1NmUxMjAifQ==
    