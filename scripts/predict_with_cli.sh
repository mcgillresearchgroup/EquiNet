equinet_path=path/to/equinet

python $chemprop_path/predict.py \
--test_path example_test.csv \
--features_path example_features.csv \
--preds_path example_output.csv \
--checkpoint_dir . \
--number_of_molecules 2 \
--num_workers 0