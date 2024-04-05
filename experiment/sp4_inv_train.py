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

import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

from core.lie_neurons_layers import *
from experiment.sp4_inv_layers import *
from data_loader.sp4_inv_data_loader import *


def init_writer(config):
    writer = SummaryWriter(
        config['log_writer_path']+"_"+str(time.localtime()), comment=config['model_description'])
    writer.add_text("train_data_path: ", config['train_data_path'])
    writer.add_text("model_save_path: ", config['model_save_path'])
    writer.add_text("log_writer_path: ", config['log_writer_path'])
    writer.add_text("shuffle: ", str(config['shuffle']))
    writer.add_text("batch_size: ", str(config['batch_size']))
    writer.add_text("init_lr: ", str(config['initial_learning_rate']))
    writer.add_text("num_epochs: ", str(config['num_epochs']))

    return writer


def test(model, test_loader, criterion, config, device):
    model.eval()
    with torch.no_grad():
        loss_sum = 0.0
        for i, sample in tqdm(enumerate(test_loader, start=0)):
            x = sample['x'].to(device)
            y = sample['y'].to(device)

            output = model(x)

            loss = criterion(output, y)
            loss_sum += loss.item()

        loss_avg = loss_sum/len(test_loader)

    return loss_avg


def train(model, train_loader, test_loader, config, device='cpu'):

    writer = init_writer(config)

    # create criterion
    criterion = nn.MSELoss().to(device)
    # criterion = nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(
    ), lr=config['initial_learning_rate'], weight_decay=config['weight_decay_rate'])
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['learning_rate_decay_rate'])
    # scheduler = optim.lr_scheduler.LinearLR(optimizer,total_iters=config['num_epochs'])

    # if config['resume_training']:
    #     checkpoint = torch.load(config['resume_model_path'])
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     start_epoch = checkpoint['epoch']
    # else:
    start_epoch = 0

    best_loss = float("inf")
    for epoch in range(start_epoch, config['num_epochs']):
        running_loss = 0.0
        loss_sum = 0.0
        model.train()
        optimizer.zero_grad()
        for i, sample in tqdm(enumerate(train_loader, start=0)):
            x = sample['x'].to(device)
            y = sample['y'].to(device)

            output = model(x)

            loss = criterion(output, y)
            loss.backward()

            # we only update the weights every config['update_every_batch'] iterations
            # This is to simulate a larger batch size
            # if (i+1) % config['update_every_batch'] == 0:
            optimizer.step()
            optimizer.zero_grad()

            # cur_training_loss_history.append(loss.item())
            running_loss += loss.item()
            loss_sum += loss.item()

            if i % config['print_freq'] == 0:
                print("epoch %d / %d, iteration %d / %d, loss: %.8f" %
                      (epoch, config['num_epochs'], i, len(train_loader), running_loss/config['print_freq']))
                running_loss = 0.0


#        scheduler.step()


        train_loss = loss_sum/len(train_loader)

        test_loss = test(
            model, test_loader, criterion, config, device)

        # log down info in tensorboard
        writer.add_scalar('training loss', train_loss, epoch)
        writer.add_scalar('test loss', test_loss, epoch)

        # if we achieve best val loss, save the model
        if test_loss < best_loss:
            best_loss = test_loss

            state = {'epoch': epoch,
                     'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'loss': train_loss,
                     'test loss': test_loss}

            torch.save(state, config['model_save_path'] +
                       '_best_test_loss_acc.pt')
        print("------------------------------")
        print("Finished epoch %d / %d, train loss: %.4f test loss: %.4f" %
              (epoch, config['num_epochs'], train_loss, test_loss))

        # save model
        state = {'epoch': epoch,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'loss': train_loss,
                 'test loss': test_loss}

        torch.save(state, config['model_save_path']+'_last_epo.pt')

    writer.close()


def main():
    # torch.autograd.set_detect_anomaly(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using ', device)

    parser = argparse.ArgumentParser(description='Train the network')
    parser.add_argument('--training_config', type=str,
                        default=os.path.dirname(os.path.abspath(__file__))+'/../config/sp4_inv/training_param.yaml')
    args = parser.parse_args()

    # load yaml file
    config = yaml.safe_load(open(args.training_config))


    training_set = sp4InvDataSet(config['train_data_path'], device=device)
    train_loader = DataLoader(dataset=training_set, batch_size=config['batch_size'],
                              shuffle=config['shuffle'])

    test_set = sp4InvDataSet(config['test_data_path'], device=device)
    test_loader = DataLoader(dataset=test_set, batch_size=config['batch_size'],
                             shuffle=config['shuffle'])


    if config['model_type'] == "LN_relu_bracket":
        model = SP4InvariantReluBracketLayers(2).to(device)
    elif config['model_type'] == "LN_relu":
        model = SP4InvariantReluLayers(2).to(device)
    elif config['model_type'] == "LN_bracket":
        model = SP4InvariantBracketLayers(2).to(device)
    elif config['model_type'] == "MLP":
        model = MLP(20).to(device)
    elif config['model_type'] == "MLP512":
        model = MLP512(20).to(device)
    elif config['model_type'] == "LN_bracket_no_residual":
        model = SP4InvariantBracketNoResidualConnectLayers(2).to(device)

    train(model, train_loader, test_loader, config, device)


if __name__ == "__main__":
    main()
