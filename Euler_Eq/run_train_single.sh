#! /bin/bash

# Run the Euler_Eq code
source venv/bin/activate

echo "Training several models for Euler Equations (Single Trajectory)"

python3 Euler_Eq/euler_poincare_eq_WithInput_train_single.py --model_type='LN_ode_WithInput6'
python3 Euler_Eq/euler_poincare_eq_WithInput_train_single.py --model_type='LN_ode_WithInput5'
python3 Euler_Eq/euler_poincare_eq_WithInput_train_single.py --model_type='LN_ode_WithInput4'
python3 Euler_Eq/euler_poincare_eq_WithInput_train_single.py --model_type='LN_ode_WithInput3'
python3 Euler_Eq/euler_poincare_eq_WithInput_train_single.py --model_type='LN_ode_WithInput2'
python3 Euler_Eq/euler_poincare_eq_WithInput_train_single.py --model_type='LN_ode_WithInput1'
python3 Euler_Eq/euler_poincare_eq_WithInput_train_single.py --model_type='LN_ode_WithInput'

python3 Euler_Eq/euler_poincare_eq_WithInput_train_single.py --model_type='neural_ode_WithInput'
python3 Euler_Eq/euler_poincare_eq_WithInput_train_single.py --model_type='neural_ode_WithInput1'
