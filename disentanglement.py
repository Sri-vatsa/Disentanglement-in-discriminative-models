import torch

# Auxiliary function
def torch_vd(d, q):
	# Compute the volume of unit l_q ball in d dimensional space
	if (q == float('inf')):
		return d * torch.log(torch.tensor(2))
	return d * torch.log(2 * torch.special.torch.exp(torch.lgamma(1+1.0/q))) - torch.lgamma(1+d*1.0/q)

def revised_ksg_estimator(variables, k=5, q=float('inf')):
	'''
	Estimate the multivariate mutual information I(X;Y;Z) = H(X) + H(Y) + H(Z) - H(X,Y,Z)
	of X, Y and Z from samples {x_i, y_i, z_i}_{i=1}^N
		
    Using Revised KSG mutual information estimator from arxiv.org/abs/1604.03006
	Input: 
    	variables: a torch tensor of size (hidden_dim, N, d) or (hidden_dim, N) where,
			hidden_dim: number of variables in MI calculation
			N: number of samples for each variable, generally corresponds to batch_size
			d: dimension of each variable, assumed to be 1 if no third dimension found.
		k: k-nearest neighbor parameter
		q: l_q norm used to decide k-nearest neighbor distance
		
    Output: a scalar representing I(variables[0];variables[1];...variables[N-1])
	'''
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if len(variables.size()) == 2:
		variables = variables[:, :, None]

	assert k <= variables.size()[1] - 1, "Set k smaller than num. samples - 1"
	hidden_dim, N, d = variables.size()
	N, d, k = torch.tensor([N, d, k]).to(device)
	data = torch.cat([variables[i] for i in range(hidden_dim)], dim=1)
 
	knn_dis = [torch.norm(data - point, dim=1, p=q).topk(k + 1, largest=False)[0][k] for point in data]
	ans_all_data = -torch.special.digamma(k) + torch.log(N) + torch_vd(d*hidden_dim, q)
	ans_individual = torch.tensor([torch.log(N) + torch_vd(d, q) for i in range(hidden_dim)]).to(device)
	
	for i in range(N):
		ans_all_data += (d * hidden_dim) * (torch.log(knn_dis[i]) / N)
		for j in range(hidden_dim):
			dist = torch.norm(variables[j] - variables[j][i], dim=1, p=q)
			num_values = torch.tensor(dist[dist < (knn_dis[i] + 1e-15)].size(dim=0) - 1).to(device)
			ans_individual[j] += -torch.log(num_values) / N \
			+ d*torch.log(knn_dis[i]) / N

	sum = torch.sum(ans_individual)
	
	return torch.abs(sum - ans_all_data)