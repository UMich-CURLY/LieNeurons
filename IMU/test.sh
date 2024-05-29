#! /bin/bash

# Run the Euler_Eq code
source venv/bin/activate

echo "Testing, only acc test"

# python3 IMU/acc_only_test.py --model_type='Acc_model_linear' --data_type='V2_02_medium'
# python3 IMU/acc_only_test.py --model_type='Acc_model_LN_1'  --data_type='V2_02_medium'
# python3 IMU/acc_only_test.py --model_type='Acc_model_mlp_1' --data_type='V2_02_medium'
# python3 IMU/acc_only_test.py --model_type='Acc_model_mlp_2' --data_type='V2_02_medium'
python3 IMU/acc_only_test.py --model_type='Acc_model_LN1_plus_Linear' --data_type='V2_02_medium'
