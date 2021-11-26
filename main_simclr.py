# Data processing
from data import get_dataloader

# Modeling
import torch
from models import *

# Plotting
import matplotlib.pyplot as plt

# Training & Evaluation
from training import train_classifier, train_encoder
from evaluation import evaluate_classifier, evaluate_simclr_encoder
from loss import regularized_loss

# Misc
import os,argparse
from datetime import datetime

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-d','--dataset', help='dataset', required=True, type=str)
parser.add_argument('-f','--finetune', help='finetune with imagenet', action='store_true')
parser.add_argument('-eo','--eval_only', help='dataset', action='store_true')
parser.add_argument('-n','--num_epochs', help='num epochs', type=int)
parser.add_argument('-bs','--batch_size', help='batch size', type=int)
parser.add_argument('-s','--start_from', help='dataset', required=True, type=int)
parser.add_argument('-lr','--lr', help='learning rate', type=float)
args = parser.parse_args()

torch.multiprocessing.set_sharing_strategy('file_system')

print('CONFIGURING')
batch_size = args.batch_size
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
fp16_precision = True
estimator = 'ksg'
data_dir = "./data/"
dataset = args.dataset # cifar10, celeba, oxfordpets, imagenet 
save_path_classifier = "./models/simclr/"+dataset+"/"+datetime.today().strftime('%Y-%m-%d')+"/"#lr_"+str(lr)+"/"
save_path_encoder = "./models/simclr/imagenet/"+datetime.today().strftime('%Y-%m-%d')+"/"#lr_"+str(lr)+"/"
path_to_pretrained_simclr = "./pretrained/"

if not os.path.exists(save_path_classifier):
    os.makedirs(save_path_classifier)
if not os.path.exists(save_path_encoder):
    os.makedirs(save_path_encoder)

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
checkpoint = torch.load(os.path.join(path_to_pretrained_simclr, 'resnet50_imagenet_bs2k_epochs600.pth.tar'))
simclr_encoder = SimCLRResNetEncoder(disentangle)
simclr_encoder.load_state_dict(checkpoint['state_dict'])

if args.finetune:
    print('FINETUNING')
    f_train_dl = get_dataloader("imagenet", "train", data_dir, batch_size=batch_size, num_workers=num_workers)
    f_test_dl = get_dataloader("imagenet", "test", data_dir, batch_size=batch_size, num_workers=num_workers)
    
    simclr_encoder, losses = train_encoder(f_train_dl, simclr_encoder, optimizer, regularized_loss, reg_lambda, estimator, num_epochs, lr, batch_size, save_path_encoder, n_views=n_views, temperature=temperature, log_interval=log_interval, fp16_precision=fp16_precision)
    
    print('EVALUATING ENCODER')
    evaluate_simclr_encoder(f_test_dl, simclr_encoder)

simclr_classifier = SimCLRResNetClassifier(simclr_encoder, num_classes, disentangle)

if not args.eval_only:
    if args.start_from != 0:
        print('LOADING CHKPT: ', str(args.start_from))
        simclr_classifier.load_state_dict(torch.load(save_path_classifier+str(args.start_from)+'.pt'))
    print('TRAINING')
    trained_resnet_model, losses = train_classifier(train_dl, simclr_classifier, args.dataset, optimizer, regularized_loss, reg_lambda, estimator, num_epochs, lr, save_path, log_interval=log_interval, pre_epoch = args.start_from)
    plt.plot([x for x in range(1, num_epochs+1)], losses, label = "Train loss")
    plt.savefig(save_path_classifier+'loss_fig.png')
    print('EVALUATING')
    evaluate_classifier(test_dl, trained_resnet_model, save_path_classifier, args.dataset)
else:
    print('EVALUATING ', save_path_classifier+str(args.start_from)+'.pt')
    if args.start_from != 0:
        simclr_classifier.load_state_dict(torch.load(save_path_classifier+str(args.start_from)+'.pt'))
    evaluate_classifier(test_dl, simclr_classifier, save_path_classifier, args.dataset)

# finetuning + training + eval args example:
# python main_simclr.py -d celeba -s 0 -lr 0.001 -n 1 -f -bs 128 
# no finetuning + training + eval args example:
# python main_simclr.py -d celeba -s 0 -lr 0.001 -n 1 -bs 128 
# no finetuning + no training + testing args example:
# python main_simclr.py -d celeba -s 0 -eo -bs 128 