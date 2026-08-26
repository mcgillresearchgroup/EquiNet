equinet_path=path/to/equinet

python $equinet_path/predict.py \
--test_path example_test.csv \
--features_path example_features.csv \
--preds_path example_output.csv \
--checkpoint_path ../equinet/pretrained_models/equinet_v0.2.0.pt \
--number_of_molecules 2 \
--num_workers 0