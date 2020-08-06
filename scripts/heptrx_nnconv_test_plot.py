import os
import os.path as osp
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (NNConv, graclus, max_pool, max_pool_x,
                                global_mean_pool)

import tqdm
import argparse

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

from datasets.hitgraphs import HitGraphDataset
from datetime import datetime

from datasets.graph import draw_sample

import awkward
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import confusion_matrix

from  mpl_toolkits import mplot3d
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap, BoundaryNorm    
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import pdb 

batch_size = 32
hidden_dim = 64
n_iters = 6
directed = False
batch_size = 1
lr = 0.01
hidden_dim = 64
n_iters = 6

from training.gnn import GNNTrainer

import logging
import pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device %s'%device)

def main(args):

    # Set paths
    directed = False
    path = osp.join(osp.dirname(os.environ['GNN_TRAINING_DATA_ROOT']), args.dataset)
    full_dataset = HitGraphDataset(path, directed=directed)
    
    #Split datasets
    fulllen = len(full_dataset)
    tv_frac = 0.10
    tv_num = math.ceil(fulllen*tv_frac)
    splits = np.cumsum([fulllen-2*tv_num,tv_num,tv_num])
    print('fulllen:', fulllen,' splits:', splits)
    
    test_dataset = torch.utils.data.Subset(full_dataset,np.arange(start=splits[0],stop=splits[1]).tolist())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    test_samples = len(test_dataset)
    d = full_dataset
    num_features = d.num_features
    
    num_classes = d[0].y.dim() if d[0].y.dim() == 1 else d[0].y.size(1)
    
    if args.categorized:
        if not args.forcecats:
            num_classes = int(d[0].y.max().item()) + 1 if d[0].y.dim() == 1 else d[0].y.size(1)
        else:
            num_classes = args.cats

    # load model definition
    the_weights = np.array([1., 1., 1., 1.]) #[0.017, 1., 1., 10.]
    
    tester = GNNTrainer(category_weights = the_weights, 
                        output_dir='./models/', device=device)

    tester.logger.setLevel(logging.DEBUG)
    strmH = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    strmH.setFormatter(formatter)
    tester.logger.addHandler(strmH)
    
    # load trained weights
    tester.build_model(name=args.model, loss_func=args.loss,
                        input_dim=num_features, hidden_dim=args.hidden_dim, n_iters=args.n_iters,
                        output_dim=num_classes)

    tester.model.load_state_dict(torch.load(args.model_weights)['model'])
    tester.print_model_summary()

    print('Model: \n%s\nParameters: %i' % (tester.model, sum(p.numel() for p in tester.model.parameters())))
    print('Testing with %s data samples'%test_samples)

    test_summary, plot_stats = tester.test(test_loader)

    del tester
        
    data, edge, true, pred , count = plot_stats 
    pdb.set_trace()
    print(test_summary)
    print('class counts')
    print(count)


    columns = ['Pred_Noise', 'Pred_Hadron', 'Pred_EM', 'Pred_MIP']
    rows    = ['True_Noise', 'True_Hadron', 'True_EM', 'True_MIP']

    for idx, item in enumerate(data):

        # pdb.set_trace()
        
        x = item[:,0]
        y = item[:,1]
        z = item[:,2]
        E = item[:,3]
        t = item[:,4]
        ytrue = true[idx]
        ypred = pred[idx]
    
        Ri = edge[idx][0]
        Ro = edge[idx][1]        
  
        # get nodeid Edge pairs
        edge_id = np.arange(len(Ri))
        incoming = [[z[n], y[n], x[n], n, e, t, p] for n, e, t, p in zip (Ri, edge_id, ytrue, ypred)]
        outgoing = [[z[n], y[n], x[n], n, e, t, p] for n, e, t, p in zip (Ro, edge_id, ytrue, ypred)]

        # sort by edge id
        sin = sorted(incoming, key=lambda x: x[4])
        sot = sorted(outgoing,  key=lambda x: x[4])

        # Creating figure 
        fig = plt.figure(figsize = (16, 10)) 
        ax1 = fig.add_subplot(221,projection='3d')   
        ax2 = fig.add_subplot(222,projection='3d') 
        ax3 = fig.add_subplot(223) 
        ax4 = fig.add_subplot(224) 

        ax1.title.set_text('True: HD:red, EM:blue, MP:green')
        ax2.title.set_text('Pred: HD:red, EM:blue, MP:green')
        ax3.title.set_text('ConfusionMatrixCounts')
        ax4.title.set_text('ConfusionMatrixPercent')
        
        ax1.set_xlabel('Z-axis', fontweight ='bold')  
        ax1.set_ylabel('Y-axis', fontweight ='bold')  
        ax1.set_zlabel('X-axis', fontweight ='bold')  
        
        ax2.set_xlabel('Z-axis', fontweight ='bold')  
        ax2.set_ylabel('Y-axis', fontweight ='bold')  
        ax2.set_zlabel('X-axis', fontweight ='bold')   

        # ax3.set_xlabel('Z-axis', fontweight ='bold')  
        # ax3.set_ylabel('Y-axis', fontweight ='bold')  
        # ax3.set_zlabel('X-axis', fontweight ='bold')  
        
        # ax4.set_xlabel('Z-axis', fontweight ='bold')  
        # ax4.set_ylabel('Y-axis', fontweight ='bold')  
        # ax4.set_zlabel('X-axis', fontweight ='bold') 

        p1s = np.array(sin[:])[:,:3]
        p2s = np.array(sot[:])[:,:3]

        cs_true = np.array(sin[:])[:,5].astype(int)
        cs_pred = np.array(sin[:])[:,6].astype(int)
        
        mt = confusion_matrix(cs_true, cs_pred)
        mtp = confusion_matrix(cs_true, cs_pred)/len(cs_pred)

        
        cdict = {0:'grey', 1:'red', 2:'blue', 3:'green'}
        
        
        #Axis 1 - all truths --------------------
        nonoise = cs_true!=0
        # HDMask = cs_true==1
        # EMMask = cs_true==2
        # MPMask = cs_true==3

        p1 = p1s[nonoise] 
        p2 = p2s[nonoise] 
        ls = np.hstack([p1,p2]).copy()  
        ls = ls.reshape((-1,2,3))  

        ax1.set_xlim(min(p1[:,0]), max(p1[:,0]))  
        ax1.set_ylim(min(p1[:,1]), max(p1[:,1]))  
        ax1.set_zlim(min(p1[:,2]), max(p1[:,2]))  
        ax2.set_xlim(min(p1[:,0]), max(p1[:,0]))  
        ax2.set_ylim(min(p1[:,1]), max(p1[:,1]))  
        ax2.set_zlim(min(p1[:,2]), max(p1[:,2]))                         

        clrs = [ cdict[int(x)] for x in cs_true[nonoise]]
        lc = Line3DCollection(ls, linewidths=0.3, colors=clrs, alpha=0.5)
        ax1.add_collection(lc)
        
        #Axis 2 - all preds -----------------------
        nonoise = cs_pred!=0
        # HDMask = cs_pred==1
        # EMMask = cs_pred==2
        # MPMask = cs_pred==3

        p1 = p1s[nonoise] 
        p2 = p2s[nonoise] 
        ls = np.hstack([p1,p2]).copy()  
        ls = ls.reshape((-1,2,3))                          

        clrs = [cdict[int(x)] for x in cs_pred[nonoise]]
        lc = Line3DCollection(ls, linewidths=0.3, colors=clrs, alpha=0.5)
        ax2.add_collection(lc)  


        colnames = []
        rownames =  []
        for i in range(4):
            if i in cs_true or i in cs_pred:
                colnames.append(columns[i])
                rownames.append(rows[i])

        
        #Axis 3 - Confusion Matrix -----------------
        ax3.axis('tight')
        ax3.axis('off')
        # pdb.set_trace()
        the_table = ax3.table(cellText=mt, colLabels=colnames, rowLabels=rownames, loc='center')

        #Axis 4 - Confusion Matrix -----------------
        ax4.axis('tight')
        ax4.axis('off')
        the_table = ax4.table(cellText=mtp, colLabels=colnames, rowLabels=rownames, loc='center')
        
        plt.savefig('./visData/testResults/'+str(idx)+'_event.pdf') 
        
        plt.close(fig)

        print(idx)

    print('Finished')
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Required positional arguments
    parser.add_argument("--model_weights", default = 'models/checkpoints/model_checkpoint_EdgeNetWithCategories_264403_5b5c05404f_csharma.best.pth.tar', help="model PyTorch state dict file [*.pth]")
    parser.add_argument("--model", default = 'EdgeNetWithCategories'  , help="Model Name")
    parser.add_argument('--dataset', '-d', default = './data/single_photon')
    parser.add_argument('--categorized', '-c', action='store_true', default=False, help='Does the model you want to train have explicit categories?')
    parser.add_argument('--forcecats', action='store_true', default=False, help='Do we want to force the number of categories?')
    parser.add_argument('--cats', default=4, type=int, help='Number of categories to force')
    parser.add_argument('--hidden_dim', default=64, type=int, help='Latent space size.')
    parser.add_argument('--n_iters', default=6, type=int, help='Number of times to iterate the graph.')
    parser.add_argument('--logger', '-g', default='./logs/ftestinglogs_EdgeNetWithCategories.txt')
    parser.add_argument('--loss', '-l', default='binary_cross_entropy', help='Loss function to use in training.')    
    
    args = parser.parse_args()
    main(args)
