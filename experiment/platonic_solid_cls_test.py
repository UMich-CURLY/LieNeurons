import sys  # nopep8
sys.path.append('.')  # nopep8

import argparse
import os
import time
import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from scipy.spatial.transform import Rotation

import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, IterableDataset

from core.lie_neurons_layers import *
from experiment.platonic_solid_cls_layers import *
from data_gen.gen_platonic_solids import *


def test_perspective(model, test_loader, criterion, config, device):
    model.eval()
    hat_layer = HatLayer(algebra_type='sl3').to(device)
    rots = random_sample_rotations(config['num_rotations'], config['rotation_factor'],device)
    with torch.no_grad():
        loss_sum = 0.0
        num_correct = 0
        num_correct_non_conj = 0
        for iter, samples in tqdm(enumerate(test_loader, start=0)):
            
            x = samples[0].to(device)
            y = samples[1].to(device)

            x_hat = hat_layer(x)
            x = rearrange(x,'b n f k -> b f k n')

            output = model(x)
            _, prediction_non_conj = torch.max(output,1)
            num_correct_non_conj += (prediction_non_conj==y).sum().item()
            for r in range(config['num_rotations']):
                cur_rot = rots[r,:,:]
                x_rot_hat = torch.matmul(cur_rot, torch.matmul(x_hat, torch.inverse(cur_rot)))
                x_rot = rearrange(vee_sl3(x_rot_hat),'b n f k -> b f k n')
                output_rot = model(x_rot)

                _, prediction = torch.max(output_rot,1)
                num_correct += (prediction==y).sum().item()
            
                loss = criterion(output_rot, y)
                loss_sum += loss.item()

        loss_avg = loss_sum/config['num_test']*config['batch_size']/config['num_rotations']
        acc_avg = num_correct/config['num_test']/config['num_rotations']
        acc_avg_non_conj = num_correct_non_conj/config['num_test']/1.0
    return loss_avg, acc_avg, acc_avg_non_conj

def random_sample_rotations(num_rotation, rotation_factor: float = 1.0, device='cpu') -> np.ndarray:
    r = np.zeros((num_rotation, 3, 3))
    for n in range(num_rotation):
        # angle_z, angle_y, angle_x
        euler = np.random.rand(3) * np.pi * 2 / rotation_factor  # (0, 2 * pi / rotation_range)
        r[n,:,:] = Rotation.from_euler('zyx', euler).as_matrix()
    return torch.from_numpy(r).type('torch.FloatTensor').to(device)




def main():
    # torch.autograd.set_detect_anomaly(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using ', device)

    parser = argparse.ArgumentParser(description='Train the network')
    parser.add_argument('--training_config', type=str,
                        default=os.path.dirname(os.path.abspath(__file__))+'/../config/platonic_solid_cls/testing_param.yaml')
    args = parser.parse_args()

    # load yaml file
    config = yaml.safe_load(open(args.training_config))


    test_set = PlatonicDataset(config['num_test'])
    test_loader = DataLoader(dataset=test_set, batch_size=config['batch_size'],
                              shuffle=config['shuffle'])
    
    if config['model_type'] == "LN_relu_bracket":
        model = LNReluBracketPlatonicSolidClassifier(3).to(device)
    elif config['model_type'] == "LN_relu":
        model = LNReluPlatonicSolidClassifier(3).to(device)
    elif config['model_type'] == "LN_bracket":
        model = LNBracketPlatonicSolidClassifier(3).to(device)
    elif config['model_type'] == "MLP":
        model = MLP(288).to(device)
    elif config['model_type'] == "LN_bracket_no_residual":
        model = LNBracketNoResidualConnectPlatonicSolidClassifier(3).to(device)
    
    print("Using model: ", config['model_type'])
    print("total number of parameters: ", sum(p.numel() for p in model.parameters()))
    
    checkpoint = torch.load(config['model_path'])
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)

    criterion = nn.CrossEntropyLoss().to(device)
    test_loss, test_acc, test_acc_non_conj = test_perspective(model, test_loader, criterion, config, device)
    print("test loss: ", test_loss)
    print("test with conjugate acc: ", test_acc)
    print("test without conjugate acc: ", "{:.4f}".format(test_acc_non_conj))

if __name__ == "__main__":
    main()
