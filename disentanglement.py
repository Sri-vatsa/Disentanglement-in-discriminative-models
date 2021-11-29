import numpy as np 
import numpy.random as nr
import torch
import torch.nn as nn
import random

# Auxiliary function
def torch_vd(d, q):
	# Compute the volume of unit l_q ball in d dimensional space
	if (q == float('inf')):
		return d * torch.log(torch.tensor(2))
	return d * torch.log(2 * torch.special.torch.exp(torch.lgamma(1+1.0/q))) - torch.lgamma(1+d*1.0/q)

def revised_ksg_estimator(variables, k=3, q=float('inf')):
	'''
	Estimate the multivariate mutual information I(X;Y;Z) = H(X) + H(Y) + H(Z) - H(X,Y,Z)
	of X, Y and Z from samples {x_i, y_i, z_i}_{i=1}^N
		
    Using Revised KSG mutual information estimator from arxiv.org/abs/1604.03006
	Input: 
    	variables: a torch tensor of size (N, hidden_dim, d) or (N, hidden_dim) where,
			N         : number of samples for each variable, generally corresponds to batch_size
			hidden_dim: number of variables in MI calculation
			d         : dimension of each variable, assumed to be 1 if no third dimension found.
		k: k-nearest neighbor parameter
		q: l_q norm used to decide k-nearest neighbor distance
		
    Output: a scalar representing I(variables[0];variables[1];...variables[N-1])
	'''
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if len(variables.size()) == 2:
		variables = variables[:, :, None]

	variables = torch.transpose(variables, 0, 1)

	assert k <= variables.size()[1] - 1, "Set k smaller than num. samples - 1"
	hidden_dim, N, d = variables.size()
	N, d, k = torch.tensor([N, d, k]).to(device)
	data = torch.cat([variables[i] for i in range(hidden_dim)], dim=1)

	knn_dis = torch.norm(data.unsqueeze(0) - data.unsqueeze(1), dim=-1, p=q).topk(k + 1, dim=1, largest=False)[0][:, k]

	ans_all_data = -torch.special.digamma(k) + torch.log(N) + torch_vd(d*hidden_dim, q)
	ans_individual = torch.full([hidden_dim], torch.log(N) + torch_vd(d, q)).to(device)
 
	ans_all_data += torch.sum((d * hidden_dim) * (torch.log(knn_dis) / N))
	
	dist = torch.norm(variables.unsqueeze(1) - variables.unsqueeze(2), dim=-1, p=q)
	num_values = torch.le(dist, (knn_dis.unsqueeze(0).unsqueeze(0) + 1e-15)).sum(dim=1).to(device) - 1
	ans_individual += (-torch.log(num_values) / N \
			+ d*torch.log(knn_dis.unsqueeze(0)) / N).sum(dim=1)

	sum = torch.sum(ans_individual)
	
	return max(sum - ans_all_data, 0)

def pop_random(lst):
    
    idx = random.randrange(0, len(lst))
    return lst.pop(idx)

def marginalPdf(values, bins, sigma, epsilon = 1e-10):
        
	residuals = values - bins.unsqueeze(0).unsqueeze(0)
	kernel_values = torch.exp(-0.5*(residuals / sigma).pow(2))
		
	pdf = torch.mean(kernel_values, dim=1)
	normalization = torch.sum(pdf, dim=1).unsqueeze(1) + epsilon
	pdf = pdf / normalization
		
	return pdf, kernel_values

	
def jointPdf(kernel_values1, kernel_values2, epsilon = 1e-10):
        
	joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2)
	normalization = torch.sum(joint_kernel_values, dim=(1,2)).view(-1, 1, 1) + epsilon
	pdf = joint_kernel_values / normalization

	return pdf

def getMutualInformation(input1, input2, bins, sigma, normalize = False, epsilon = 1e-10):
	'''
	input1: B, C, hidden_dim
	input2: B, C, hidden_dim
	return: scalar
	'''
        
	B, C, hidden_dim = input1.shape
	assert((input1.shape == input2.shape))

	x1 = input1.view(B, hidden_dim, C)
	x2 = input2.view(B, hidden_dim, C)
	pdf_x1, kernel_values1 = marginalPdf(x1, bins, sigma, epsilon)
	pdf_x2, kernel_values2 = marginalPdf(x2, bins, sigma, epsilon)
        
	pdf_x1x2 = jointPdf(kernel_values1, kernel_values2, epsilon)
        
	H_x1 = -torch.sum(pdf_x1*torch.log2(pdf_x1 + epsilon), dim=1)
	H_x2 = -torch.sum(pdf_x2*torch.log2(pdf_x2 + epsilon), dim=1)
	H_x1x2 = -torch.sum(pdf_x1x2*torch.log2(pdf_x1x2 + epsilon), dim=(1,2))

	mutual_information = H_x1 + H_x2 - H_x1x2
		
	if normalize:
		mutual_information = 2*mutual_information/(H_x1+H_x2)

	return mutual_information

def sample_pairwise_activations(activations, sigma = sigma, normalize = normalize, epsilon = epsilon, bins = bins):
    '''
    Inputs:
    activations: a torch tensor of size (B, hidden_dim, d) or (B, hidden_dim) where,
      B: number of samples for each variable, generally corresponds to batch_size
			hidden_dim: number of variables in MI calculation
			d: dimension of each variable, assumed to be 1 if no third dimension found.
    '''
    num_hidden_nodes = activations.shape[1]
    pairs = []
		lst = list(range(num_hidden_nodes))
		while lst:
    	rand1 = pop_random(lst)
    	rand2 = pop_random(lst)
    	pair = rand1, rand2
    	pairs.append(pair)

		final_mi = 0.0
		for i in range(len(pairs)):
			input1 = activations[:, pairs[i][0], :]
			input1 = input1.unsqueeze(1)
	 		input2 = activations[:, pairs[i][1], :]
			input2 = input2.unsqueeze(1)
			final_mi += getMutualInformation(input1, input2, bins, sigma)
	 
		final_mi /= len(pairs)
		return final_mi