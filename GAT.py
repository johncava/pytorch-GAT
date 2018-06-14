import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utility import *

# Toy graph structure as a symmetric graph
graph = [[1,0,0,0,0,0,0,0,0,0],
         [0,1,0,0,0,0,0,0,0,0],
         [1,0,1,0,0,0,0,0,0,0],
         [0,1,1,1,0,0,0,0,0,0],
         [1,1,0,0,1,0,0,0,0,0],
         [0,0,1,0,0,1,0,0,0,0],
         [1,1,0,0,0,0,1,0,0,0],
         [0,0,1,0,1,1,0,1,0,0],
         [1,0,0,1,0,1,0,0,1,0],
         [0,0,0,1,0,0,1,0,0,1]]

# Toy label
label = [[1,1],[1,1],[1,1],[1,1],[1,1],[0,0],[0,0],[0,0],[0,0],[0,0]]

# Turn array into numpy array
graph = np.array(graph)

# Turn symmetric graph into an adjacency graph
graph = graph + graph.T - np.eye(10)

# Random feature matrix for the graph
features = np.random.rand(10,10) * 10

# Turn features into pytorch Variable
features = Variable(torch.Tensor(features))

# Turn label into pytorch Variable
label = Variable(torch.Tensor(label))

# Get neighbors for attention model
neighbors = get_neighbors(graph)

# Define W_out which would be equal in this case to the number of features of the label dataset => 2
W_out = 2

# Define Graph Attention Model
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        # Linear Function that takes the features (h_i) and turns it into new features (new_h_i)
        self.W = nn.Linear(10,W_out)
        # Note: Attention Mechanism takes twice the output of Linear Function (W) because of the concatentation of Wh_i and Wh_j (Wh_i || Wh_j)
        self.a = nn.Linear(2*W_out,1)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self,x):
        # List to hold the new h_i values calculated from the attention mechanism
        new_h_list = []
        # Go through each node and perform attention in respect to its neighbors (which has been computed previously)
        for primary_index,primary_node in enumerate(neighbors):
            h = []
            W_hjs = []
            e = torch.Tensor([])
            # Reference Equation (1),(3) : e_ij = a(Wh_i, Wh_j) = Leaky_Relu(attention(Wh_i, Wh_j)) => Neural_Network( Wh_i || Wh_j )
            for neighbor in primary_node:
                # Neighbor node features matrix multiplied with W. Also stored for future use when multiplying against alphas in line 75
                W_hj = self.W(features[neighbor])
                # Note: concatenation of e_ij into a single torch tensor such that there is one line to do F.softmax(e) in line 70
                e = torch.cat((e,self.leaky_relu(self.a(torch.cat((self.W(x[primary_index]),W_hj))))))
                W_hjs.append(W_hj)
            # Softmax(e_ij) Reference: Equation (2)
            a = F.softmax(e)
            # Reference: Equation (4)
            new_h = torch.Tensor([0.0]*W_out)
            for a_ij, w_hj in zip(a,W_hjs):
                new_h += a_ij * w_hj
            new_h_list.append(F.leaky_relu(new_h))
            ######################################
        return torch.stack(new_h_list)

# Initialize Attention Model
attention = Attention()

loss_function = nn.MSELoss()
optimizer = optim.Adam(attention.parameters(), lr=1e-3)
max_iterations = 10

for iteration in xrange(max_iterations):
    prediction = attention(features)
    optimizer.zero_grad()
    loss = loss_function(prediction,label)
    print loss.item()
    loss.backward()
    optimizer.step()

print "Done"