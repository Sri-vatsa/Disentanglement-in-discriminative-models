
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler, autocast

import tqdm
import logging 
from torch.utils.tensorboard import SummaryWriter

from utils import accuracy
from loss import info_nce_loss

import os
from os.path import exists

def train_classifier(dataloader, model, dataset, optimizer, loss_func, reg_lambda, estimator, num_epochs, learning_rate, save_path, log_interval=100, pre_epoch=0):
    opt = optimizer(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
    model.to(device)
    model.train()
    epoch_losses = []
    for epoch in tqdm(range(num_epochs), position=0, leave=True):  # loop over the dataset multiple times
        epoch += pre_epoch
        running_loss = 0.0
        step = 0
        for i, batchdata in enumerate(tqdm(dataloader, position=0, leave=True)):
            if dataset == 'celeba':
                batchdata, names, target = batchdata["image"], batchdata["image_name"], batchdata["attributes"]
                batchdata, target = batchdata.cuda(), target.cuda()
                inputs, labels = batchdata, target
            else:
                inputs, labels = batchdata
            inputs = inputs.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            outputs, representation = model(inputs)
            dis_dict = {'state': model.module.disentangle, 'estimator': None, 'activations': None}
            if dis_dict['state']:
                dis_dict['activations'] = representation
                dis_dict['estimator'] = estimator
            loss = loss_func(outputs, labels, dis_dict, lambda_=reg_lambda, celeba=True if dataset=='celeba' else False)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            step += 1
        epoch_loss = running_loss / step
        epoch_losses.append(epoch_loss)
        # print statistics
        if epoch % log_interval == log_interval-1: 
            print('[Epoch %d] loss: %.3f' % (epoch + 1, epoch_loss))
            if not exists(save_path+str(epoch+1)+'.pt'):
                open(save_path+str(epoch+1)+'.pt', 'x')
            try:
                torch.save(model.module.state_dict(), save_path+str(epoch+1)+'.pt')
            except:
                print(str(epoch)+' could not save!')
    print('Completed Training')
    return model, epoch_losses

def train_encoder(dataloader, model, optimizer, loss_func, reg_lambda, estimator, num_epochs, learning_rate, batch_size, save_path, n_views=2, temperature=0.07, log_interval=100, fp16_precision=False):
  opt = optimizer(model.parameters(), lr=learning_rate)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  model.train()
  epoch_losses = []
  writer = SummaryWriter()
  logging.basicConfig(filename=os.path.join(writer.log_dir, 'training.log'), level=logging.DEBUG)
  scaler = GradScaler(enabled=fp16_precision)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(dataloader), eta_min=0,
                                                           last_epoch=-1)
  logging.info(f"Start SimCLR training for {num_epochs} epochs.")

  for epoch in tqdm(range(num_epochs), position=0, leave=True):  # loop over the dataset multiple times

      running_loss = 0.0
      step = 0
      for i, batchdata in enumerate(tqdm(dataloader, position=0, leave=True)):

          inputs, _ = batchdata
          inputs = torch.cat(inputs, dim=0)
          inputs = inputs.to(device)

          opt.zero_grad()

          with autocast(enabled=fp16_precision):
            outputs, representation = model(inputs)
            logits, labels =  info_nce_loss(outputs, batch_size=batch_size, n_views=n_views, temperature=temperature)

            dis_dict = {'state': model.disentangle, 'estimator': None, 'activations': None}
            if dis_dict['state']:
              dis_dict['activations'] = representation
              dis_dict['estimator'] = estimator

            # Need to pass the disentanglement layer activations to get MI based regularizer loss
            loss = loss_func(outputs, labels, dis_dict, lambda_=reg_lambda)
            top1 = accuracy(logits, labels, topk=(1,))
          scaler.scale(loss).backward()
          scaler.step(opt)
          scaler.update()

          running_loss += loss.item()
          step += 1

      epoch_loss = running_loss / step
      epoch_losses.append(epoch_loss)
      # print statistics
      if epoch % log_interval == log_interval-1: 
            print('[Epoch %d] loss: %.3f' % (epoch + 1, epoch_loss))
            if not exists(save_path+str(epoch+1)+'.pt'):
                open(save_path+str(epoch+1)+'.pt', 'x')
            try:
                torch.save(model.module.state_dict(), save_path+str(epoch+1)+'.pt')
            except:
                print(str(epoch)+' could not save!')
      # warmup for the first 10 epochs
      if epoch >= 10:
          scheduler.step()
      
      logging.debug(f"Epoch: {epoch}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

  logging.info("Training has finished.")
  return model, epoch_losses