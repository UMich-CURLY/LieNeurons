#! /bin/bash

# Run the Euler_Eq code
source venv/bin/activate

echo "Training IMU, only acc test"

# python3 IMU/acc_only.py --model_type='Acc_model_linear'
# python3 IMU/acc_only.py --model_type='Acc_model_LN_1'
# python3 IMU/acc_only.py --model_type='Acc_model_mlp_1'
# python3 IMU/acc_only.py --model_type='Acc_model_mlp_2'
python3 IMU/acc_only.py --model_type='Acc_model_LN1_plus_Linear'
