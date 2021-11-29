import torch
import torch.nn as nn
import torch.nn.functional as F

from disentanglement import revised_ksg_estimator, sample_pairwise_activations

def regularized_loss(outputs, targets, dis_dict, lambda_=0.5, celeba=False):
    if celeba:
        cross_entropy = F.binary_cross_entropy_with_logits(outputs, targets.float().unsqueeze(1))
    else:
        cross_entropy = nn.CrossEntropyLoss()(outputs, targets)
    regularization = 0

    if dis_dict['state'] and dis_dict['estimator'] == 'ksg':
        regularization = revised_ksg_estimator(dis_dict['activations'])
        
    elif dis_dict['state'] and dis_dict['estimator'] == 'bgk':
        sigma = 2*0.4**2
        num_bins = 256
        normalize = False
        epsilon = 1e-10
        bins = nn.Parameter(torch.linspace(0, 255, num_bins).float(), requires_grad=False)
        regularization = sample_pairwise_activations(dis_dict['activations'], sigma = sigma, normalize = normalize, epsilon = epsilon, bins = bins)

    loss = cross_entropy + lambda_*regularization
    return loss

# https://github.com/sthalles/SimCLR/blob/master/simclr.py
def info_nce_loss(features, batch_size, n_views=2, temperature=0.07):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #print(batch_size)
        #print([torch.arange(batch_size) for i in range(int(n_views))])
        labels = torch.cat([torch.arange(batch_size) for i in range(int(n_views))], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (n_views *batch_size, n_views * batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        logits = logits / temperature
        return logits, labels