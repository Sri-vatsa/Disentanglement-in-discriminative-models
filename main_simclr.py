# Data processing
from data import get_dataloader

# Modeling
import torch
from models import *

# Plotting
import matplotlib.pyplot as plt

# Training & Evaluation
from training import train_classifier, train_encoder
from evaluation import evaluate_classifier
from loss import regularized_loss

# Misc
import os,argparse
from datetime import datetime

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-d','--dataset', help='dataset', required=True, type=str)
parser.add_argument('-f','--finetune', help='finetune with imagenet', required=True, type=str)
parser.add_argument('-eo','--eval_only', help='dataset', action='store_true')
parser.add_argument('-n','--num_epochs', help='num epochs', type=int)
parser.add_argument('-s','--start_from', help='dataset', required=True, type=int)
parser.add_argument('-lr','--lr', help='learning rate', type=float)
args = parser.parse_args()

torch.multiprocessing.set_sharing_strategy('file_system')

print('CONFIGURING')
batch_size = 32
num_workers = 4
num_epochs = args.num_epochs
lr = args.lr
reg_lambda = 0.5
log_interval = 1 # Num epochs after which loss is reported
optimizer = torch.optim.Adam
n_views = 2
is_contrastive = False
temperature = 0.07
disentangle = False
estimator = 'ksg'
data_dir = "./data/"
dataset = args.dataset # cifar10, celeba, oxfordpets, imagenet 
save_path = "./models/"+dataset+"/"+datetime.today().strftime('%Y-%m-%d')+"/"#lr_"+str(lr)+"/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
train_dl = get_dataloader(dataset, "train", data_dir, batch_size=batch_size, num_workers=num_workers)
test_dl = get_dataloader(dataset, "test", data_dir, batch_size=batch_size, num_workers=num_workers)

print('BUILDING')
if args.dataset=='celeba':
    num_classes = train_dl.dataset.dataset.__num_classes__()
    assert num_classes == 1
elif args.dataset=='oxfordpets':
    num_classes = len(train_dl.dataset.dataset.classes)
else:
    num_classes = len(train_dl.dataset.classes)

print('num classes: ',str(num_classes))
checkpoint = torch.load('./pretrained_models/resnet50_imagenet_bs512_epochs100.pth.tar')
simclr_encoder = SimCLRResNetEncoder(disentangle)
simclr_encoder.load_state_dict(checkpoint['state_dict'])
simclr_classifier = SimCLRResNetClassifier(simclr_encoder, num_classes, disentangle)

if not args.eval_only and not args.finetune:
    #simclr_encoder, losses = train_encoder(train_dl, simclr_encoder, optimizer, regularized_loss, reg_lambda, estimator, num_epochs, lr, batch_size, n_views=n_views, temperature=temperature, log_interval=log_interval, fp16_precision=fp16_precision)
    pass #TODO

elif not args.eval_only and not args.finetune:
    if args.start_from != 0:
        print('LOADING CHKPT: ', str(args.start_from))
        simclr_classifier.load_state_dict(torch.load(save_path+str(args.start_from)+'.pt'))
    print('TRAINING')
    trained_resnet_model, losses = train_classifier(train_dl, simclr_classifier, args.dataset, optimizer, regularized_loss, reg_lambda, estimator, num_epochs, lr, save_path, log_interval=log_interval, pre_epoch = args.start_from)
    plt.plot([x for x in range(1, num_epochs+1)], losses, label = "Train loss")
    plt.savefig(save_path+'loss_fig.png')
    print('EVALUATING')
    evaluate_classifier(test_dl, trained_resnet_model, save_path, args.dataset)
else:
    print('EVALUATING ', save_path+str(args.start_from)+'.pt')
    if args.start_from != 0:
        simclr_classifier.load_state_dict(torch.load(save_path+str(args.start_from)+'.pt'))
    evaluate_classifier(test_dl, simclr_classifier, save_path, args.dataset)
    
# training args example:
# python train_resnet.py -d celeba -s 0 -lr 0.001 -n 1
# testing args example:
# python train_resnet.py -d celeba -s 0 -eo