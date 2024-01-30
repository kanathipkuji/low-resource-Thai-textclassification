dataset_name='tscc'
text='filtered_fact'
label='label'
csv_sep=','
shot='100'

python scripts/stats/compute-processed-dataset-stats.py \
    --train_dir "./data/processed/$dataset_name/$dataset_name-$shot/train" \
    --valid_dir "./data/processed/$dataset_name/$dataset_name-$shot/valid" \
    --test_dir "./data/processed/$dataset_name/$dataset_name-$shot/test" \
    --output_dir ./data/stats/ \
    --dataset_name $dataset_name-$shot \
    --text_column_name $text \
    --label_column_name $label \
    --csv_sep "$csv_sep" \