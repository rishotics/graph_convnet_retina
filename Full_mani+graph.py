import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import pdb #pdb.set_trace()
import collections
import time
import numpy as np

import sys
sys.path.insert(0, 'lib/')


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if torch.cuda.is_available():
    print('cuda available')
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
    torch.cuda.manual_seed(1)
else:
    print('cuda not available')
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor
    torch.manual_seed(1)


from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from sklearn.preprocessing import normalize
import csv
from sklearn.model_selection import train_test_split
import pandas as pd
h=pd.read_csv("Abnormal.csv")
h=np.array(h)
print(h.shape)

a=pd.read_csv("Healthy.csv")
a=np.array(a)
print(a.shape)

x=np.concatenate((h,a),axis=0)
print(x.shape)
M,N=x.shape
y=np.ones((M,))

y[0:h.shape[0]]=0
y[h.shape[0]:M]=1

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
#mnist = input_data.read_data_sets('datasets', one_hot=False) # load data in folder datasets/
m,n=X_test.shape
ym,=y_test.shape
train_data=(X_train)
print(int(m))
val_data = (X_test[0:int(m/2),:])
test_data = (X_test[int(m/2):int(m),:])
train_labels = y_train
val_labels = y_test[0:int(m/2),]
test_labels =y_test[int(m/2):int(m),]
print(train_data.shape)
print(train_data[0:30,0:30])
print(train_labels.shape)
print(val_data.shape)
print(val_labels.shape)
print(test_data.shape)
print(test_labels.shape)

from grid_graph import grid_graph
from coarsening import coarsen
from coarsening import lmax_L
from coarsening import perm_data
from coarsening import rescale_L

# Construct graph



# Delete existing network if exists

d=[1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e+0,1e+1,1e+2,1e+3,1e+4,1e+5,1e+6]
print(d)
a=[]
# loop over epochs
for decay in d:

    print("l2=%.8f" %(decay))
    t_start = time.time()
    grid_side = 28
    number_edges = 8
    metric = 'euclidean'
    A = grid_graph(grid_side,number_edges,metric) # create graph of Euclidean grid

    # Compute coarsened graphs
    coarsening_levels = 5

    L, perm = coarsen(A, coarsening_levels)

    # Compute max eigenvalue of graph Laplacians
    lmax = []
    for i in range(coarsening_levels):
        lmax.append(lmax_L(L[i]))
    print('lmax: ' + str([lmax[i] for i in range(coarsening_levels)]))

    # Reindex nodes to satisfy a binary tree structure
    train_data = perm_data(train_data, perm)
    val_data = perm_data(val_data, perm)
    test_data = perm_data(test_data, perm)

    print(train_data.shape)
    print(val_data.shape)
    print(test_data.shape)

    print('Execution time: {:.2f}s'.format(time.time() - t_start))
    del perm

    class my_sparse_mm(torch.autograd.Function):
        """
        Implementation of a new autograd function for sparse variables,
        called "my_sparse_mm", by subclassing torch.autograd.Function
        and implementing the forward and backward passes.
        """

        def forward(self, W, x):  # W is SPARSE
            self.save_for_backward(W, x)
            y = torch.mm(W, x)
            return y

        def backward(self, grad_output):
            W, x = self.saved_tensors
            grad_input = grad_output.clone()
            grad_input_dL_dW = torch.mm(grad_input, x.t())
            grad_input_dL_dx = torch.mm(W.t(), grad_input )
            return grad_input_dL_dW, grad_input_dL_dx


    class Graph_ConvNet_LeNet5(nn.Module):

        def __init__(self, net_parameters):

            print('Graph ConvNet: LeNet5')

            super(Graph_ConvNet_LeNet5, self).__init__()

            # parameters
            D, CL1_F, CL1_K, CL2_F, CL2_K, FC1_F, FC2_F = net_parameters
            FC1Fin = CL2_F*(D//16)

            # graph CL1
            self.cl1 = nn.Linear(CL1_K, CL1_F)
            Fin = CL1_K; Fout = CL1_F;
            scale = np.sqrt( 2.0/ (Fin+Fout) )
            self.cl1.weight.data.uniform_(-scale, scale)
            self.cl1.bias.data.fill_(0.0)
            self.CL1_K = CL1_K; self.CL1_F = CL1_F;

            # graph CL2
            self.cl2 = nn.Linear(CL2_K*CL1_F, CL2_F)
            Fin = CL2_K*CL1_F; Fout = CL2_F;
            scale = np.sqrt( 2.0/ (Fin+Fout) )
            self.cl2.weight.data.uniform_(-scale, scale)
            self.cl2.bias.data.fill_(0.0)
            self.CL2_K = CL2_K; self.CL2_F = CL2_F;

            # FC1
            self.fc1 = nn.Linear(FC1Fin, FC1_F)
            Fin = FC1Fin; Fout = FC1_F;
            scale = np.sqrt( 2.0/ (Fin+Fout) )
            self.fc1.weight.data.uniform_(-scale, scale)
            self.fc1.bias.data.fill_(0.0)
            self.FC1Fin = FC1Fin

            # FC2
            self.fc2 = nn.Linear(FC1_F, FC2_F)
            Fin = FC1_F; Fout = FC2_F;
            scale = np.sqrt( 2.0/ (Fin+Fout) )
            self.fc2.weight.data.uniform_(-scale, scale)
            self.fc2.bias.data.fill_(0.0)

            # nb of parameters
            nb_param = CL1_K* CL1_F + CL1_F          # CL1
            nb_param += CL2_K* CL1_F* CL2_F + CL2_F  # CL2
            nb_param += FC1Fin* FC1_F + FC1_F        # FC1
            nb_param += FC1_F* FC2_F + FC2_F         # FC2
            print('nb of parameters=',nb_param,'\n')


        def init_weights(self, W, Fin, Fout):

            scale = np.sqrt( 2.0/ (Fin+Fout) )
            W.uniform_(-scale, scale)

            return W


        def graph_conv_cheby(self, x, cl, L, lmax, Fout, K):

            # parameters
            # B = batch size
            # V = nb vertices
            # Fin = nb input features
            # Fout = nb output features
            # K = Chebyshev order & support size
            B, V, Fin = x.size(); B, V, Fin = int(B), int(V), int(Fin)

            # rescale Laplacian
            lmax = lmax_L(L)
            L = rescale_L(L, lmax)

            # convert scipy sparse matric L to pytorch
            L = L.tocoo()
            indices = np.column_stack((L.row, L.col)).T
            indices = indices.astype(np.int64)
            indices = torch.from_numpy(indices)
            indices = indices.type(torch.LongTensor)
            L_data = L.data.astype(np.float32)
            L_data = torch.from_numpy(L_data)
            L_data = L_data.type(torch.FloatTensor)
            L = torch.sparse.FloatTensor(indices, L_data, torch.Size(L.shape))
            L = Variable( L , requires_grad=False)
            if torch.cuda.is_available():
                L = L.cuda()

            # transform to Chebyshev basis
            x0 = x.permute(1,2,0).contiguous()  # V x Fin x B
            x0 = x0.view([V, Fin*B])            # V x Fin*B
            x = x0.unsqueeze(0)                 # 1 x V x Fin*B

            def concat(x, x_):
                x_ = x_.unsqueeze(0)            # 1 x V x Fin*B
                return torch.cat((x, x_), 0)    # K x V x Fin*B

            if K > 1:
                x1 = my_sparse_mm()(L,x0)              # V x Fin*B
                x = torch.cat((x, x1.unsqueeze(0)),0)  # 2 x V x Fin*B
            for k in range(2, K):
                x2 = 2 * my_sparse_mm()(L,x1) - x0
                x = torch.cat((x, x2.unsqueeze(0)),0)  # M x Fin*B
                x0, x1 = x1, x2

            x = x.view([K, V, Fin, B])           # K x V x Fin x B
            x = x.permute(3,1,2,0).contiguous()  # B x V x Fin x K
            x = x.view([B*V, Fin*K])             # B*V x Fin*K

            # Compose linearly Fin features to get Fout features
            x = cl(x)                            # B*V x Fout
            x = x.view([B, V, Fout])             # B x V x Fout

            return x


        # Max pooling of size p. Must be a power of 2.
        def graph_max_pool(self, x, p):
            if p > 1:
                x = x.permute(0,2,1).contiguous()  # x = B x F x V
                x = nn.MaxPool1d(p)(x)             # B x F x V/p
                x = x.permute(0,2,1).contiguous()  # x = B x V/p x F
                return x
            else:
                return x


        def forward(self, x, d, L, lmax):

            # graph CL1
            x = x.unsqueeze(2) # B x V x Fin=1
            x = self.graph_conv_cheby(x, self.cl1, L[0], lmax[0], self.CL1_F, self.CL1_K)
            x = F.relu(x)
            x = self.graph_max_pool(x, 4)

            # graph CL2
            x = self.graph_conv_cheby(x, self.cl2, L[2], lmax[2], self.CL2_F, self.CL2_K)
            x = F.relu(x)
            x = self.graph_max_pool(x, 4)

            # FC1
            x = x.view(-1, self.FC1Fin)
            x = self.fc1(x)
            x = F.relu(x)
            x  = nn.Dropout(d)(x)

            # FC2
            x = self.fc2(x)

            return x


        def loss(self, y, y_target, l2_regularization):

            loss = nn.CrossEntropyLoss()(y,y_target)

            l2_loss = 0.0
            for param in self.parameters():
                data = param* param
                l2_loss += data.sum()

            loss += 0.5* l2_regularization* l2_loss

            return loss


        def update(self, lr):

            update = torch.optim.SGD( self.parameters(), lr=lr, momentum=0.9 )

            return update


        def update_learning_rate(self, optimizer, lr):

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            return optimizer


        def evaluation(self, y_predicted, test_l):

            _, class_predicted = torch.max(y_predicted.data, 1)
            return 100.0* (class_predicted == test_l).sum()/ y_predicted.size(0)
    try:
        del net
        print('Delete existing network\n')
    except NameError:
        print('No existing network to delete\n')



    # network parameters
    D = train_data.shape[1]
    CL1_F = 32
    CL1_K = 25
    CL2_F = 64
    CL2_K = 25
    FC1_F = 512
    FC2_F = 10
    net_parameters = [D, CL1_F, CL1_K, CL2_F, CL2_K, FC1_F, FC2_F]


    # instantiate the object net of the class
    net = Graph_ConvNet_LeNet5(net_parameters)
    if torch.cuda.is_available():
        net.cuda()
    print(net)


    # Weights
    L_net = list(net.parameters())


    # learning parameters
    learning_rate = 0.05
    dropout_value = 0.3
    l2_regularization = decay
    batch_size = 100
    num_epochs = 30
    train_size = train_data.shape[0]
    nb_iter = int(num_epochs * train_size) // batch_size
    print('num_epochs=',num_epochs,', train_size=',train_size,', nb_iter=',nb_iter)


    # Optimizer
    global_lr = learning_rate
    global_step = 0
    decay = 0.95
    decay_steps = train_size
    lr = learning_rate
    optimizer = net.update(lr)
    import matplotlib.pyplot as plt
    indices = collections.deque()
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        # reshuffle
        indices.extend(np.random.permutation(train_size)) # rand permutation

        # reset time
        t_start = time.time()

        # extract batches
        running_loss = 0.0
        running_accuray = 0
        running_total = 0
        while len(indices) >= batch_size:

            # extract batches
            batch_idx = [indices.popleft() for i in range(batch_size)]
            train_x, train_y = train_data[batch_idx,:], train_labels[batch_idx]
            train_x = Variable( torch.FloatTensor(train_x).type(dtypeFloat) , requires_grad=False)
            train_y = train_y.astype(np.int64)
            train_y = torch.LongTensor(train_y).type(dtypeLong)
            train_y = Variable( train_y , requires_grad=False)

            # Forward
            y = net.forward(train_x, dropout_value, L, lmax)
            loss = net.loss(y,train_y,l2_regularization)
            loss_train = loss.data

            # Accuracy
            acc_train = net.evaluation(y,train_y.data)

            # backward
            loss.backward()

            # Update
            global_step += batch_size # to update learning rate
            optimizer.step()
            optimizer.zero_grad()

            # loss, accuracy
            running_loss += loss_train
            running_accuray += acc_train
            running_total += 1

            # print
            if not running_total%100: # print every x mini-batches
                print('epoch= %d, i= %4d, loss(batch)= %.4f, accuray(batch)= %.2f' % (epoch+1, running_total, loss_train, acc_train))


        # print
        t_stop = time.time() - t_start
        print('epoch= %d, loss(train)= %.3f, accuracy(train)= %.3f, time= %.3f, lr= %.5f' %
              (epoch+1, running_loss/running_total, running_accuray/running_total, t_stop, lr))



        # update learning rate
        lr = global_lr * pow( decay , float(global_step// decay_steps) )
        optimizer = net.update_learning_rate(optimizer, lr)


        # Test set
        running_accuray_test = 0
        running_total_test = 0
        indices_test = collections.deque()
        indices_test.extend(range(test_data.shape[0]))
        t_start_test = time.time()
        while len(indices_test) >= batch_size:
            batch_idx_test = [indices_test.popleft() for i in range(batch_size)]
            test_x, test_y = test_data[batch_idx_test,:], test_labels[batch_idx_test]
            test_x = Variable( torch.FloatTensor(test_x).type(dtypeFloat) , requires_grad=False)
            y = net.forward(test_x, 0.0, L, lmax)
            test_y = test_y.astype(np.int64)
            test_y = torch.LongTensor(test_y).type(dtypeLong)
            test_y = Variable( test_y , requires_grad=False)
            acc_test = net.evaluation(y,test_y.data)
            running_accuray_test += acc_test
            running_total_test += 1
        t_stop_test = time.time() - t_start_test
        print('  accuracy(test) = %.3f %%, time= %.3f' % (running_accuray_test / running_total_test, t_stop_test))
        a.append(running_accuray_test / running_total_test)


    L, perm = coarsen(A, coarsening_levels)

    # Compute max eigenvalue of graph Laplacians
    lmax = []
    for i in range(coarsening_levels):
        lmax.append(lmax_L(L[i]))
    print('lmax: ' + str([lmax[i] for i in range(coarsening_levels)]))

    # Reindex nodes to satisfy a binary tree structure
    train_data = perm_data(train_data, perm)
    val_data = perm_data(val_data, perm)
    test_data = perm_data(test_data, perm)

    print(train_data.shape)
    print(val_data.shape)
    print(test_data.shape)

    print('Execution time: {:.2f}s'.format(time.time() - t_start))
    del perm

print(a)
plt.plot(a)
