import scipy.spatial as ss
import scipy.stats as sst
from scipy.special import digamma,gamma
from sklearn.neighbors import KernelDensity
from cvxopt import matrix,solvers

from math import log,pi,exp
import numpy as np

#Auxilary functions
def vd(d,q):
	# Compute the volume of unit l_q ball in d dimensional space
	if (q==float('inf')):
		return d*log(2)
	return d*log(2*gamma(1+1.0/q)) - log(gamma(1+d*1.0/q))

def entropy(x,k=5,q=float('inf')):
	# Estimator of (differential entropy) of X 
	# Using k-nearest neighbor methods 
	assert k <= len(x)-1, "Set k smaller than num. samples - 1"
	N = len(x)
	d = len(x[0])	
	thre = 3*(log(N)**2/N)**(1/d)
	tree = ss.cKDTree(x)
	knn_dis = [tree.query(point,k+1,p=q)[0][k] for point in x]
	truncated_knn_dis = [knn_dis[s] for s in range(N) if knn_dis[s] < thre]
	ans = -digamma(k) + digamma(N) + vd(d,q)
	return ans + d*np.mean(map(log,knn_dis))

def kde_entropy(x):
	# Estimator of (differential entropy) of X 
	# Using resubstitution of KDE
	N = len(x)
	d = len(x[0])
	local_est = np.zeros(N)
	for i in range(N):
		kernel = sst.gaussian_kde(x.transpose())
		local_est[i] = kernel.evaluate(x[i].transpose())
	return -np.mean(map(log,local_est))

def revised_ksg_estimator(variables, k=5, q=float('inf')):
	'''
		Estimate the multivariate mutual information I(X;Y;Z) = H(X) + H(Y) + H(Z) - H(X,Y,Z)
		of X, Y and Z from samples {x_i, y_i, z_i}_{i=1}^N
		
    Using Revised KSG mutual information estimator from arxiv.org/abs/1604.03006
		Input: 
    variables: a numpy array of (hidden_dim, N, d) shape where,
			hidden_dim: number of variables in MI calculation
			N: number of samples for each variable, generally corresponds to batch_size
			d: dimension of each variable, could be 1.
		k: k-nearest neighbor parameter
		q: l_q norm used to decide k-nearest neighbor distance
		
    Output: a scalar representing I(variables[0];variables[1];...variables[N-1])
	'''
	
	assert k <= variables.shape[1] - 1, "Set k smaller than num. samples - 1"
	
	hidden_dim, N, d = variables.shape

	data = np.concatenate([variables[i] for i in range(hidden_dim)], axis=1)

	tree_all_data = ss.cKDTree(data)
	tree_individual = [ss.cKDTree(variables[i]) for i in range(hidden_dim)]

	knn_dis = [tree_all_data.query(point, k + 1, p=q)[0][k] for point in data]
		
	ans_all_data = -digamma(k) + log(N) + vd(d*hidden_dim, q)
	ans_individual = [log(N) + vd(d, q) for i in range(hidden_dim)]
	
	for i in range(N):
		ans_all_data += (d * hidden_dim) * (log(knn_dis[i]) / N)
		for j in range(hidden_dim):
			ans_individual[j] += -log(len(tree_individual[j].query_ball_point(variables[j][i], knn_dis[i] + 1e-15, p=q)) - 1) / N \
			+ d*log(knn_dis[i]) / N

	sum = 0
	for ans in ans_individual:
		sum += ans
	
	return sum - ans_all_data