dataset_name='wongnai'
shot='100'
label='star'
csv_sep=';'

python scripts/split-data.py \
    --input_path "./data/processed/$dataset_name/$dataset_name-cleaned.csv" \
    --output_dir "./data/processed/$dataset_name/$dataset_name-$shot/" \
    --label_column_name "$label" \
    --csv_sep "$csv_sep" \
    --shot "$shot" \
    --random_state 123