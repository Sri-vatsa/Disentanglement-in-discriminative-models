import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import tqdm
import numpy as np

# Metrics
from sklearn import metrics
from sklearn.metrics import roc_auc_score

import json
import matplotlib.pyplot as plt

from loss import info_nce_loss
from utils import accuracy, plot_confusion_matrix_2, unsigned_correlation_coefficient

def evaluate_simclr_encoder(dataloader, model, batch_size=32, n_views=2, temperature=0.07):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  model.eval()
  top1_accuracy = 0
  top5_accuracy = 0
  with torch.no_grad():
    for i, batchdata in enumerate(tqdm(dataloader, position=0, leave=True)):
      inputs, labels = batchdata
      inputs = torch.cat(inputs, dim=0)
      inputs = inputs.to(device)

      outputs, _ = model(inputs)
      logits, labels =  info_nce_loss(outputs, batch_size=batch_size, n_views=n_views, temperature=temperature)
      top1, top5 = accuracy(logits, labels, topk=(1,5))
      top1_accuracy += top1[0]
      top5_accuracy += top5[0]
  
    top1_accuracy /= (i + 1)
    top5_accuracy /= (i + 1)
  print("Top-1 accuracy: {}, Top-5 accuracy: {}".format(top1_accuracy, top5_accuracy))
  
def evaluate_disentanglement(dataloader, model, save_path, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_correlation_scores = []
    with torch.no_grad():
        for i, batchdata in enumerate(tqdm(dataloader, position=0, leave=True)):
            if dataset == 'celeba':
                batchdata, names, target = batchdata["image"], batchdata["image_name"], batchdata["attributes"]
                inputs, labels = batchdata.cuda(), target
            else:
                inputs, labels = batchdata
            inputs = inputs.to(device)
            outputs = model(inputs)
            if dataset == 'celeba':
                outputs = (torch.sigmoid(outputs[0])>= 0.5).type(torch.int32).squeeze().cpu()
            else:
                outputs = outputs[0].cpu().numpy().argmax(axis=1)
            labels = labels.numpy()
            corr_score_b = unsigned_correlation_coefficient(outputs[1])
            all_correlation_scores.append(corr_score_b)

        corr_score = sum(all_correlation_scores)/len(all_correlation_scores) if len(all_correlation_scores) != 0 else None
        print("correlation score: {}".format(corr_score))
        scores = {'corelation': corr_score}
        with open(save_path+'corr-score.json', 'w') as f:
            json.dump(scores, f)

def evaluate_classifier(dataloader, model, save_path, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for i, batchdata in enumerate(tqdm(dataloader, position=0, leave=True)):
            if dataset == 'celeba':
                batchdata, names, target = batchdata["image"], batchdata["image_name"], batchdata["attributes"]
                inputs, labels = batchdata.cuda(), target
            else:
                inputs, labels = batchdata
            inputs = inputs.to(device)
            outputs = model(inputs)
            if dataset == 'celeba':
                outputs = (torch.sigmoid(outputs[0])>= 0.5).type(torch.int32).squeeze().cpu()
            else:
                outputs = outputs[0].cpu().numpy().argmax(axis=1)
            labels = labels.numpy()
            all_preds.append(outputs)
            all_labels.append(labels)
        preds_ = np.concatenate(all_preds)
        labels_ = np.concatenate(all_labels)
        acc_score  = metrics.accuracy_score(preds_, labels_)
        f1_score = metrics.f1_score(preds_, labels_, average='macro')
        precision_score = metrics.precision_score(preds_, labels_, average='macro')
        recall_score = metrics.recall_score(preds_, labels_, average='macro')
        print("acc: {}".format(acc_score))
        print("f1_score: {}".format(f1_score))
        print("precision_score: {}".format(precision_score))
        print("recall_score: {}".format(recall_score))
        scores = {'acc': acc_score,
                  'f1_score': f1_score,
                  'precision_score': precision_score,
                  'recall_score': recall_score}
        if dataset=='celeba':
            classes = ['Smiling', 'Not_Smiling']
        elif dataset=='oxfordpets':
            classes = dataloader.dataset.dataset.classes
        else:
            classes = dataloader.dataset.classes
        cm = metrics.confusion_matrix(labels_, preds_)
        plot_confusion_matrix_2(cm, classes)
        plt.savefig(save_path+'eval_cm.png')
        with open(save_path+'results.json', 'w') as f:
            json.dump(scores, f)