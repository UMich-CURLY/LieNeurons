# IMU_dynamics

<!-- Nerual ODE with inputs -->

## Run

1. **Install dependencies**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r Requirments.txt
    ```

2. **Generate data first by running**

    ```bash
    python3 data_gen/gen_Euler_eq_data.py
    ```

3. **Run the training script**

    ```bash
    ./Euler_Eq/run_train_multiple.sh
    ```

4. **Run the testing script**

    ```bash
    ./Euler_Eq/run_test_multiple.sh
    ```










