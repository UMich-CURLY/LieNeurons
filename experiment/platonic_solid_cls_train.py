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


def init_writer(config):
    writer = SummaryWriter(
        config['log_writer_path']+"_"+str(time.localtime()), comment=config['model_description'])
    # writer.add_text("train_data_path: ", config['train_data_path'])
    writer.add_text("model_save_path: ", config['model_save_path'])
    writer.add_text("log_writer_path: ", config['log_writer_path'])
    writer.add_text("shuffle: ", str(config['shuffle']))
    writer.add_text("batch_size: ", str(config['batch_size']))
    writer.add_text("init_lr: ", str(config['initial_learning_rate']))
    writer.add_text("num_train: ", str(config['num_train']))

    return writer

def random_sample_rotations(num_rotation, rotation_factor: float = 1.0, device='cpu') -> np.ndarray:
    r = np.zeros((num_rotation, 3, 3))
    for n in range(num_rotation):
        # angle_z, angle_y, angle_x
        euler = np.random.rand(3) * np.pi * 2 / rotation_factor  # (0, 2 * pi / rotation_range)
        r[n,:,:] = Rotation.from_euler('zyx', euler).as_matrix()
    return torch.from_numpy(r).type('torch.FloatTensor').to(device)

def test(model, test_loader, criterion, config, device):
    model.eval()
    with torch.no_grad():
        loss_sum = 0.0
        num_correct = 0
        for iter, samples in tqdm(enumerate(test_loader, start=0)):
            
            x = samples[0].to(device)
            y = samples[1].to(device)

            x = rearrange(x,'b n f k -> b f k n')
            output = model(x)

            _, prediction = torch.max(output,1)
            num_correct += (prediction==y).sum().item()
            
            loss = criterion(output, y)
            loss_sum += loss.item()

        loss_avg = loss_sum/config['num_test']*config['batch_size']
        acc_avg = num_correct/config['num_test']

    return loss_avg, acc_avg


def train(model, train_loader, test_loader, config, device='cpu'):

    writer = init_writer(config)

    # create criterion
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(
    ), lr=config['initial_learning_rate'], weight_decay=config['weight_decay_rate'])
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['learning_rate_decay_rate'])
    # scheduler = optim.lr_scheduler.LinearLR(optimizer,total_iters=config['num_epochs'])

    if config['resume_training']:
        checkpoint = torch.load(config['resume_model_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint['epoch']
    else:
        start_iter = 0

    best_loss = float("inf")
    best_acc = 0.0
    running_loss = 0.0
    loss_sum = 0.0
    hat_layer = HatLayer(algebra_type='sl3').to(device)

    for iter, samples in tqdm(enumerate(train_loader, start=0)):
        
        model.train()
        optimizer.zero_grad()

        x = samples[0].to(device)
        cls = samples[1].to(device)
        if config['train_augmentation']:
            rot = random_sample_rotations(1, config['rotation_factor'],device)
            x_hat = hat_layer(x)
            x_rot_hat = torch.matmul(rot, torch.matmul(x_hat, torch.inverse(rot)))
            x = vee_sl3(x_rot_hat)

        x = rearrange(x,'b n f k -> b f k n')

        output = model(x)
        # print(output)
        loss = criterion(output, cls)
        loss.backward()

        # we only update the weights every config['update_every_batch'] iterations
        # This is to simulate a larger batch size
        # if (i+1) % config['update_every_batch'] == 0:
        optimizer.step()
        optimizer.zero_grad()

        # cur_training_loss_history.append(loss.item())
        running_loss += loss.item()
        loss_sum += loss.item()

        # if iter % config['print_freq'] == 0:
        #     print("iteration %d / %d, loss: %.8f" %
        #             (iter, config['num_train'], running_loss/config['print_freq']))
        #     running_loss = 0.0


#        scheduler.step()

        # train_top1, train_top5, _ = validate(train_loader, model, criterion, config, device)

        train_loss = loss_sum/(iter+1)

        test_loss, test_acc = test(
            model, test_loader, criterion, config, device)

        # log down info in tensorboard
        writer.add_scalar('training loss', train_loss, iter)
        writer.add_scalar('test loss', test_loss, iter)
        writer.add_scalar('test acc', test_acc, iter)

        # if we achieve best val loss, save the model
        if test_loss < best_loss:
            best_loss = test_loss

            state = {'iter': iter,
                     'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'loss': train_loss,
                     'test loss': test_loss,
                     'test acc': test_acc}

            torch.save(state, config['model_save_path'] +
                       '_best_test_loss.pt')
            
        if test_acc > best_acc:
            best_acc = test_acc

            state = {'iter': iter,
                     'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'loss': train_loss,
                     'test loss': test_loss,
                     'test acc': test_acc}

            torch.save(state, config['model_save_path'] +
                       '_best_test_acc.pt')
        print("------------------------------")
        # print("Finished epoch %d / %d, training top 1 acc: %.4f, training top 5 acc: %.4f, \
        #       validation top1 acc: %.4f, validation top 5 acc: %.4f" %\
        #     (epoch, config['num_epochs'], train_top1, train_top5, val_top1, val_top5))
        print("Finished iteration %d / %d, train loss: %.4f test loss: %.4f test acc: %.4f" %
              (iter, config['num_train']/config['batch_size'], train_loss, test_loss, test_acc))

    # save model
    state = {'iter': iter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'test loss': test_loss}

    torch.save(state, config['model_save_path']+'_last_iter.pt')

    writer.close()


def main():
    # torch.autograd.set_detect_anomaly(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using ', device)

    parser = argparse.ArgumentParser(description='Train the network')
    parser.add_argument('--training_config', type=str,
                        default=os.path.dirname(os.path.abspath(__file__))+'/../config/platonic_solid_cls/training_param.yaml')
    args = parser.parse_args()

    # load yaml file
    config = yaml.safe_load(open(args.training_config))

    # create dataset and dataloader
    training_set = PlatonicDataset(config['num_train'])
    train_loader = DataLoader(dataset=training_set, batch_size=config['batch_size'],
                              shuffle=config['shuffle'])

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

    train(model, train_loader, test_loader, config, device)


if __name__ == "__main__":
    main()
