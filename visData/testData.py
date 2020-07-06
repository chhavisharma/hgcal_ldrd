import numpy as np
import scipy as scp
import torch

import pdb

if __name__ == "__main__":
    print('Testing torch import')
    print('Exiting')
    data = np.load('./../../data/raw/partGun_PDGid15_x1000_Pt3.0To100.0_NTUP_2_hgcal_graph_pos_evt71.npz')
    
    print(data)
    keys = [k for k in data.keys()]

    print('Keys',keys)
    for k in keys:
        print(k,':',data[k].shape)
    
    c=0
    for example in data[keys[0]]:
        print(example)
        c=c+1
        if(c>2):
            break

    X = data[keys[0]]
    pdb.set_trace()



