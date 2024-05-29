import argparse

# Step 2: Create the argument parser
parser = argparse.ArgumentParser('Only Acc Test')

# Step 3: Add arguments
parser.add_argument('--model_type', type=str, default='unspecified', help='Specify the type of model to use')
parser.add_argument('--data_type', type=str, default='unspecified', help='Specify the type of data to use')

# Step 4: Parse the arguments
args = parser.parse_args()

# Step 5: Use the parsed arguments
print(f"Model Type: {args.model_type}")
print(f"Data Type: {args.data_type}")
