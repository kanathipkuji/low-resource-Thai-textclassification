dataset_name='wongnai'
shot='5'
text='review'
label='star'
csv_sep=';'
vib='True'
neptune_api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxZDIyNDdmMi05YTY3LTRmYTktODk3OC05ZmMxZmU1NmUxMjAifQ=='

python scripts/train-finetuner.py \
    --train_dir "./data/processed/$dataset_name/$dataset_name-$shot/train" \
    --valid_dir "./data/processed/$dataset_name/$dataset_name-$shot/valid" \
    --test_dir "./data/processed/$dataset_name/$dataset_name-$shot/test" \
    --text_column_name $text \
    --label_column_name $label \
    --csv_sep "$csv_sep" \
    --deterministic F \
    --output_dir "./results/$dataset_name-$shot-vib-$vib" \
    --logging_dir "./results/$dataset_name-$shot-vib-$vib-logs" \
    --logging_steps 5 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --num_train_epochs 50 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --warmup_steps 500 \
    --weight_decay 0.01 \
    --neptune_project kanathip137/tscc-finetuner \
    --neptune_api_token $neptune_api_token \
    --run_tags "$dataset_name" "$shot-shot" \
    --ib $vib \
    --ib_dim 384 \
    --beta 1e-05 \
    --kl_annealing linear \
    