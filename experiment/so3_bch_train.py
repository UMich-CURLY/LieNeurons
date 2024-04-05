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
from experiment.so3_bch_layers import *
from data_loader.so3_bch_data_loader import *
from experiment.so3_bch_test import test_bch


def frobenius_norm_loss(x,y,z):
    # || exp^x @ exp^y @ exp^-z - I ||_f
    diff = torch.matmul(torch.matmul(exp_so3(x), exp_so3(y)), exp_so3(-z)) - torch.eye(3).to(x.device)
    frobenius_norm = torch.norm(diff, p='fro')
    return frobenius_norm



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


def test_frobenius(model, test_loader, criterion, config, device):
    model.eval()
    hat_so3 = HatLayer(algebra_type='so3').to(device)
    with torch.no_grad():
        loss_sum = 0.0
        for i, sample in tqdm(enumerate(test_loader, start=0)):
            x = sample['x'].to(device)
            y = sample['y'].to(device)

            output = model(x)

            loss = frobenius_norm_loss(hat_so3(x[:,0,:].squeeze(-1)), hat_so3(x[:,1,:].squeeze(-1)), hat_so3(output))
            loss_sum += loss.item()

        loss_avg = loss_sum/len(test_loader)

    return loss_avg

def train(model, train_loader, test_loader,test_loader_1, config, device='cpu'):

    writer = init_writer(config)

    # create criterion
    criterion = nn.MSELoss().to(device)
    # criterion = nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(
    ), lr=config['initial_learning_rate'], weight_decay=config['weight_decay_rate'])
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['learning_rate_decay_rate'])
    # scheduler = optim.lr_scheduler.LinearLR(optimizer,total_iters=config['num_epochs'])

    if config['resume_training']:
        print("resume training from ", config['resume_model_path'])
        checkpoint = torch.load(config['resume_model_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        
        print("start from epoch ", start_epoch)
        print("loss: ",checkpoint['loss'])
        print("test loss: ", checkpoint['test loss'])
        # print(model.model.network[0].linear.weight)
        # print(model.model.network[0].bilinear.bi_params)
    else:
        start_epoch = 0

    hat_so3 = HatLayer(algebra_type='so3').to(device)

    best_loss = float("inf")
    for epoch in range(start_epoch, config['num_epochs']):
        running_loss = 0.0
        loss_sum = 0.0
        model.train()
        optimizer.zero_grad()
        for i, sample in tqdm(enumerate(train_loader, start=0)):
            x = sample['x'].to(device)
            y = sample['y'].to(device)
            # print("=================================================")
            # print("i ",i)
            # for name, param in model.named_parameters():
            #     print(name, param.data)

            output = model(x)
            
            
            # print("x",x.shape)
            # print("hat so3",hat_so3(x[:,0,:].squeeze(-1)).shape)  
            # loss = criterion(output, y)
            loss = frobenius_norm_loss(hat_so3(x[:,0,:].squeeze(-1)), hat_so3(x[:,1,:].squeeze(-1)), hat_so3(output))
            loss.backward()

            # we only update the weights every config['update_every_batch'] iterations
            # This is to simulate a larger batch size
            # if (i+1) % config['update_every_batch'] == 0:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
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

        # train_top1, train_top5, _ = validate(train_loader, model, criterion, config, device)

        train_loss = loss_sum/len(train_loader)

        test_loss = test_frobenius(
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
            
            # print(model.model.network[0].linear.weight)
            
            # print(model.model.network[0].bilinear.bi_params)


            # Temporary testing because we're unable to save emlp model correctly
            if config['full_eval_during_training']:
                test_config = yaml.safe_load(open(os.path.dirname(os.path.abspath(__file__))+'/../config/so3_bch/testing_param.yaml'))
                error_fro_avg, error_log_avg, error_fro_conj_avg, error_log_conj_avg, diff_output_avg = test_bch(model, test_loader_1, test_config, device)
                print("error fro avg: ", error_fro_avg)
                print("error log avg: ", error_log_avg)
                print("error fro conj avg: ", error_fro_conj_avg)
                print("error log conj avg: ", error_log_conj_avg)
                print("avg diff output: ", diff_output_avg)
        # print("test_loss type:",type(test_loss_equiv))
        # # print("avg diff output type: ", diff_output_avg.dtype)
        # print("test with augmentation loss: ", test_loss_equiv)
        

        print("------------------------------")
        # print("Finished epoch %d / %d, training top 1 acc: %.4f, training top 5 acc: %.4f, \
        #       validation top1 acc: %.4f, validation top 5 acc: %.4f" %\
        #     (epoch, config['num_epochs'], train_top1, train_top5, val_top1, val_top5))
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
                        default=os.path.dirname(os.path.abspath(__file__))+'/../config/so3_bch/training_param.yaml')
    args = parser.parse_args()

    # load yaml file
    config = yaml.safe_load(open(args.training_config))

    training_set = so3BchDataSet(
        config['train_data_path'], device=device)
    train_loader = DataLoader(dataset=training_set, batch_size=config['batch_size'],
                              shuffle=config['shuffle'])

    test_set = so3BchTestDataSet(
        config['test_data_path'], device=device)
    test_loader = DataLoader(dataset=test_set, batch_size=config['batch_size'],
                             shuffle=config['shuffle'])
    test_loader_1 = DataLoader(dataset=test_set, batch_size=1,
                                shuffle=config['shuffle'])

    if config['model_type'] == "LN_relu_bracket":
        model = SO3EquivariantReluBracketLayers(2).to(device)
    elif config['model_type'] == "LN_relu":
        model = SO3EquivariantReluLayers(2).to(device)
    elif config['model_type'] == "LN_bracket":
        model = SO3EquivariantBracketLayers(2).to(device)
    elif config['model_type'] == "MLP":
        model = MLP(6).to(device)
    elif config['model_type'] == "EMLP":
        import emlp.nn.pytorch as emlpnn
        from emlp.reps import T,V
        from emlp.groups import SO
        G = SO(3)
        reps = 2*V(G)
        reps_out = V(G)
        class EMLPModel(nn.Module):
            def __init__(self):
                super(EMLPModel, self).__init__()
                self.model = emlpnn.EMLP(reps, reps_out, group=G, num_layers=3,ch=128)

            def forward(self, x):
                B,_,_,_ = x.shape
                x = torch.reshape(x, (B, -1))
                return self.model(x)
            
        model = EMLPModel().to(device)

    elif config['model_type'] == "e3nn_norm":
        import e3nn
        from experiment.so3_bch_baseline_layers import E3nnMLPNorm
        model = E3nnMLPNorm(invariant=False).to(device)
    elif config['model_type'] == "e3nn_s2grid":
        import e3nn
        from experiment.so3_bch_baseline_layers import E3nnMLPS2Grid
        model = E3nnMLPS2Grid(invariant=False).to(device)
        
    elif config['model_type'] == "VN_relu":
        model = SO3EquivariantVNReluLayers(2).to(device)
    # elif config['model_type'] == "LN_bracket_no_residual":
    #     model = SO3EquivariantBracketNoResidualConnectLayers(2).to(device)
        
    print("total number of parameters: ", sum(p.numel() for p in model.parameters()))
    # for name, param in model.named_parameters():
    #     print(name, param.shape)

    train(model, train_loader, test_loader,test_loader_1, config, device)


if __name__ == "__main__":
    main()
