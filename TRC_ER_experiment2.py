
import numpy as np
import networkx as nx
from numpy.random import default_rng
rng = default_rng(1)
import cvxpy as cp
import random
import math
from sklearn.metrics import matthews_corrcoef, f1_score
import argparse 

parser = argparse.ArgumentParser(description="L1 Optimization with external epsilon value")
parser.add_argument('--epsilon', type=float, required=True, help="Epsilon value to control the sparsity")
args = parser.parse_args()

rng = default_rng(1)

#parameters
N = 1000
T = 2
epsilon = args.epsilon
prob = np.round((np.log(N)/N)*(1-epsilon),5)
seed = 1

r_values = [1.2, 2.6, 3, 3.8 ]
np.random.seed(42)
random_numbers = np.round(np.random.rand(N,N),2)
alpha = np.triu(random_numbers) +np.triu(random_numbers).T -np.diag(np.diag(np.triu(random_numbers)))
r_vector = np.sort([r_values[np.random.randint(0,len(r_values))] for p in range(0, N)]) 

def network(N, r_vector, T, x0, alpha):
    G = nx.erdos_renyi_graph(N, prob, seed=1, directed=False)
    G = G.to_directed() 
    A = (nx.adjacency_matrix(G).toarray())
    
    x = np.zeros([T,N])
    x[0,:] = x0
    coupling = np.zeros([T,N])
    
    for k in range(T-1):
        for i in range(N):
            for j in range(N):
                if j != i:
                    coupling[k,i] += A.T[i,j]*alpha[i,j]*(x[k,i]-x[k,j])
                    
            x[k+1,i] = r_vector[i] * x[k,i] * (1 - x[k,i]) +coupling[k,i]     
            
    return x

def signal(N, r_vector, T, alpha):
    signal_matrix = np.zeros([T, N, N])
    initial_conditions = np.zeros((N, N))
    for i in range(N):
        x0 = np.zeros(N)  
        x0[i] = np.random.uniform(0.5, 1)  
        initial_conditions[i, :] = x0  
    
    for i in range(N):
        x0 = initial_conditions[i, :]  
        signal_matrix[:, :, i] = network(N, r_vector, T, x0, alpha)
    
    return signal_matrix

ww = signal(N, r_vector, T, alpha)

def l_1_optimization(y, PHI, solver='ECOS', verbose=False):
    n, m = PHI.shape
    x = cp.Variable(m)
    objective = cp.Minimize(cp.norm(x, 1))
    constraints = [PHI @ x == y]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=verbose)
    return x.value, problem.value


precision = [10**(-9),10**(-10)]

meanrates = np.arange(4,21)
predictions = np.zeros([N, len(meanrates), len(precision), N])

for i, meanrate in enumerate(meanrates):
    for k, prec in enumerate(precision):
        M = rng.normal(loc=0, scale=(1 / int((N * (meanrate / 100)))), size=(int((N * (meanrate / 100))), N))
        
        for q in range(N):
            Y = np.dot(M, ww[1, :, q])  
            
            x_t, nnz_l1 = l_1_optimization(Y, M)
            
            predictions[:, i, k, q] = x_t

            predictions[:, i, k, q] = np.where(np.abs(predictions[:, i, k, q]) < prec, 0, predictions[:, i, k, q])


sparse_x = ww[1, :, :].flatten()

mcc_scores = np.zeros([len(meanrates), len(precision)])

for i, meanrate in enumerate(meanrates):
    for k, prec in enumerate(precision):
        supp_of_gt = (sparse_x != 0).astype(int)  
        supp_of_exp = (predictions[:, i, k].flatten() != 0).astype(int)  
        mcc_scores[i, k] = matthews_corrcoef(supp_of_gt, supp_of_exp)

filename = f'ERex2_mcc_scores_N{N}_prob{str(prob).replace(".", "")}.txt'

np.savetxt(filename, mcc_scores)
