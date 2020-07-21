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

    test_summary = tester.test(test_loader)

    # test_loss, test_acc, test_eff, test_fp, test_fn, test_pur = test(model, test_loader, test_samples)

    # print('Testing: Loss: {:.4f}, Eff.: {:.4f}, FalsePos: {:.4f}, FalseNeg: {:.4f}, Purity: {:,.4f}'.format(test_loss, test_eff, test_fp, test_fn, test_pur))

    # with open(args.logger, 'a+') as f:
    #     f.write(str(datetime.now()))
    #     for k in train_summary.keys():
    #         f.write('\n'+k+'\n')
    #         f.write(str(train_summary[k]))
    #         f.write('\n')

    # # plotting:
    # figs = []
    # t = tqdm.tqdm(enumerate(test_loader),total=test_samples/batch_size)
    # out = []
    # y = []
    # x = []
    # edge_index = []
    # simmatched = []
    # for i,data in t:
    #     x.append(data.x.cpu().detach().numpy())
    #     y.append(data.y.cpu().detach().numpy())
    #     edge_index.append(data.edge_index.cpu().detach().numpy())
    #     simmatched.append(data.simmatched.cpu().detach().numpy())
    #     data = data.to(device)

    # out = awkward.fromiter(out)
    # x = awkward.fromiter(x)
    # y = awkward.fromiter(y)
    # simmatched = awkward.fromiter(simmatched)
    # edge_index = awkward.fromiter(edge_index)

    # predicted_edge = (out > 0.5)
    # truth_edge = (y > 0.5)
    # node_energy = x[:,:,4]
    # node_layer = x[:,:,2]

    # predicted_connected_node_indices = awkward.JaggedArray.concatenate([edge_index[:,0][predicted_edge], edge_index[:,1][predicted_edge]], axis=1)
    # predicted_connected_node_indices = awkward.fromiter(map(np.unique, predicted_connected_node_indices))
    # predicted_energy_sum = node_energy[predicted_connected_node_indices].sum()
    # truth_connected_node_indices = awkward.JaggedArray.concatenate([edge_index[:,0][truth_edge],edge_index[:,1][truth_edge]], axis=1)
    # truth_connected_node_indices = awkward.fromiter(map(np.unique, truth_connected_node_indices))
    # truth_energy_sum = node_energy[truth_connected_node_indices].sum()
    
    # nonzeromask = (truth_energy_sum !=0.0)
    # energy_captured_ratio = predicted_energy_sum[nonzeromask]/truth_energy_sum[nonzeromask]
    
    # fig,axes = plt.subplots(figsize=(12, 7))
    # _, bins,_ = axes.hist(energy_captured_ratio, bins=100)
    # plt.title("Ratio of energy sum for predicted hits/truth (preprocessed) hits")
    # plt.ylabel("events (pos+neg)")
    # plt.xlabel("Ratio")
    # cut = energy_captured_ratio[(energy_captured_ratio>0.975) & (energy_captured_ratio < 1.025)] #NB: restricted fit
    # (mu, sigma) = stats.norm.fit(cut)
    # c_paras = stats.crystalball.fit(cut)
    # lnspc = np.linspace(bins[0], bins[-1], len(bins))
    # pdf_g = stats.norm.pdf(lnspc, mu, sigma)
    # pdf_c = stats.crystalball.pdf(lnspc, *c_paras)
    # pdf_g = pdf_g / pdf_g.sum() * len(energy_captured_ratio)
    # pdf_c = pdf_c / pdf_c.sum() * len(energy_captured_ratio)
    # plt.plot(lnspc, pdf_g, label="Norm, restricted fit")
    # plt.plot(lnspc, pdf_c, label="Crystalball, restricted fit")
    # plt.legend(loc='upper left')
    # figs.append(fig)


    # # visualisation
    # idxs = [0]
    # for idx in idxs:
    #     fig = draw_sample(x[idx].regular(), edge_index[idx].regular()[0], edge_index[idx].regular()[1], y[idx], out[idx], sim_list=simmatched[idx])
    #     figs.append(fig)
    
    # import matplotlib.backends.backend_pdf
    # pdf = matplotlib.backends.backend_pdf.PdfPages("test_plots.pdf")
    # for fig in figs: 
    #     pdf.savefig(fig)
    # pdf.close()

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
