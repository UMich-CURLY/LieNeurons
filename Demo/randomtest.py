

x = 6.6
y = 0.3

print(x % y)
print(x // y)

a = "figures/"
b = "True_Tra.figures"
# b = None
c = a + b
print(c)

a = 5.0
b = -2.0

c = round(a/b)
print(c)

import torch

if torch.cuda.is_available():
    device = torch.device('cuda:' + str(0))
    print('Using GPU')
else:
    device = torch.device('cpu')
    print('Using CPU')
    


import os

model_type = "LN_ode_WithInput1"
model_save_path = "weights/Euler_WithInput/Single_Tra/"+model_type + "/"
fig_save_path = "figures/Euler_WithInput/Single_Tra/"+model_type + "/"

log_writer_path = "logs/Euler_WithInput/Single_Tra/"+model_type + "/"

if os.path.exists(model_save_path + model_type + '_best_val_loss_acc.pt'):
    best_loss = torch.load(model_save_path + model_type + '_best_val_loss_acc.pt')['loss']
    print("model exists, best loss: ", best_loss)
else:
    best_loss = float('inf')
    print("model does not exist, initialize best loss")

def sin_or_cos_chosen():
        if torch.rand(1) > 0.5:
            print("sin")
            return torch.sin
        else:
            print("cos")
            return torch.cos
        
for i in range(10):
    func_test = sin_or_cos_chosen()
    print(func_test(torch.tensor([0.0])))
    
import inspect
import pickle

# Define your lambda expression
my_lambda = lambda x: x * 2

# Save the lambda expression and its source code
source_code = inspect.getsource(my_lambda).strip()
print("Lambda expression:", source_code)
# with open("lambda_source.pkl", "wb") as f:
#     pickle.dump((my_lambda, source_code), f)

# # Later, to reload and view the lambda expression
# with open("lambda_source.pkl", "rb") as f:
#     reloaded_lambda, reloaded_source_code = pickle.load(f)

# print("Reloaded lambda expression:")
# print(reloaded_source_code)

my_list = [1, 2, 3, 4, 5, 6, 7, 8]
last_two = my_list[-2:]
print(last_two)
temp = my_list[-2]
print(temp)

testlist = []
aaa = torch.tensor([1.0])
testlist.append(aaa)
aaa = torch.tensor([2.0])
testlist.append(aaa)
aaa = torch.tensor([3.0])
testlist.append(aaa)

temp = torch.stack(testlist[-2:], dim=0)
print(temp)

aaa = torch.tensor([4.0, 5.0, 6.0])
temp = aaa[-2:]
print(temp)


import yaml

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            # Load the YAML content
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
            return None

# data = yaml.safe_load("data/MH_02_easy/imu0/sensor.yaml")
data = read_yaml_file("data/MH_02_easy/mav0/imu0/sensor.yaml")

fre  = int(data['rate_hz'])
print(fre)
print(f"rate_hz: {data['rate_hz']} (type: {type(data['rate_hz'])})")

import pandas as pd

df = pd.read_csv("data/MH_02_easy/mav0/imu0/data.csv",header=1, names=["time", "w_x", "w_y", "w_z", "a_x", "a_y", "a_z"])
# Convert the DataFrame to a NumPy array
print(df.columns)
numpy_array = df.to_numpy()
print("numpy_array[0,:]:", numpy_array[0,:])
print("numpy_array[1,:]:", numpy_array[1,:])
print("numpy_array.shape:", numpy_array.shape)
print(df.head())


print("df.iloc[0]:", df.iloc[0])
print("df.iloc[1]:", df.iloc[1])

print("df['time'][0]:", df['time'][0])

print("df[time] type:",df['time'].dtype)
print("df[wx] type:",df['w_x'].dtype)


# Iterate through the DataFrame
# for index, row in data_frame.iterrows():
#     img_path = row['image_path']
#     label = row['label']

a = torch.tensor([1.0, 2.0, 3.0])
b = a.unsqueeze(0)
print("a shape:", a.shape)
print("b shape:", b.shape)
c = a.unsqueeze_(0)
print("a shape:", a.shape)
print("c shape:", c.shape)


