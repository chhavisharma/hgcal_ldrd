import os
import numpy as np
import scipy as scp
import torch

from  mpl_toolkits import mplot3d
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap, BoundaryNorm    
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import pdb

def sparse_to_graph(X, Ri_rows, Ri_cols, Ro_rows, Ro_cols, y, dtype=np.float32):
    n_nodes, n_edges = X.shape[0], Ri_rows.shape[0]
    spRi_idxs = np.stack([Ri_rows.astype(np.int64), Ri_cols.astype(np.int64)])
    # Ri_rows and Ri_cols have the same shape
    spRi_vals = np.ones((Ri_rows.shape[0],), dtype=dtype)
    spRi = (spRi_idxs,spRi_vals,n_nodes,n_edges)

    spRo_idxs = np.stack([Ro_rows.astype(np.int64), Ro_cols.astype(np.int64)])
    # Ro_rows and Ro_cols have the same shape
    spRo_vals = np.ones((Ro_rows.shape[0],), dtype=dtype)
    spRo = (spRo_idxs,spRo_vals,n_nodes,n_edges)

    if y.dtype != np.uint8:
        y = y.astype(np.uint8)

    return spRi, spRo, y

if __name__ == "__main__":

    ''' Read Files '''
    events = []
    root = './../../data/raw/'
    for file in os.listdir(root):
        if file.endswith(".npz"):
            path = os.path.join(root, file)
            events.append(path)

    for itr,event_path in enumerate(events):

        if(itr==20):
            break

        print('event:',itr)
        print('path: ',event_path)

        item  = np.load(event_path)

        keys = [k for k in item.keys()]
        for k in keys: 
            print(k,':',item[k].shape)

        x = item['X'][:,0]
        y = item['X'][:,1]
        z = item['X'][:,2]
        E = item['X'][:,3]
        t = item['X'][:,4]
        
        ytruth = item['y']

        Ri_rows = item['Ri_rows']
        Ri_cols = item['Ri_cols']
        Ro_rows = item['Ro_rows']        
        Ro_cols = item['Ro_cols']

        # spRi, spRo, ytruth = sparse_to_graph(x, Ri_rows, Ri_cols, Ro_rows, Ro_cols, ytruth)
        # print((Ri_rows==Ro_rows).all())
        # print((Ri_cols==Ro_cols).all())
        # print(len(Ri_rows), len(Ri_cols), len(Ro_rows), len(Ro_cols))
        # print(min(Ri_rows), min(Ri_cols), min(Ro_rows), min(Ro_cols))
        # print(max(Ri_rows), max(Ri_cols), max(Ro_rows), max(Ro_cols))

        # get nodeid Edge pairs
        incoming = [[z[n], y[n], x[n], n,e,c] for n, e, c in zip (Ri_rows, Ri_cols, ytruth)]
        outgoing = [[z[n], y[n], x[n], n,e,c] for n, e, c in zip (Ro_rows, Ro_cols, ytruth)]
        
        # sort by edge id
        sin = sorted(incoming, key=lambda x: x[4])
        sot = sorted(outgoing,  key=lambda x: x[4])

        # Creating figure 
        fig = plt.figure(figsize = (20, 9)) 
        ax1 = fig.add_subplot(121,projection='3d')   
        ax2 = fig.add_subplot(122,projection='3d')   
                
        #Axis 1 - hits 
        ax1.set_xlabel('Z-axis', fontweight ='bold')  
        ax1.set_ylabel('Y-axis', fontweight ='bold')  
        ax1.set_zlabel('X-axis', fontweight ='bold')  
        ax1.scatter3D(z, y, x, s=1, color = "green")  

        # Axis 2 - Edges
        ax2.set_xlabel('Z-axis', fontweight ='bold')  
        ax2.set_ylabel('Y-axis', fontweight ='bold')  
        ax2.set_zlabel('X-axis', fontweight ='bold')   

        p1s = np.array(sin[:])[:,:3]
        p2s = np.array(sot[:])[:,:3]
        cs = np.array(sin[:])[:,5]
        
        nonoise = cs!=0.0
        cdict = {0:'grey', 1:'red', 2:'blue', 3:'green'}
        clrs = [ cdict[int(x)] for x in cs[nonoise]]
        # pdb.set_trace()

        p1 = p1s[nonoise] 
        p2 = p2s[nonoise] 
        ls = np.hstack([p1,p2]).copy()  
        ls = ls.reshape((-1,2,3))  

        ax2.set_xlim(min(p1[:,0]), max(p1[:,0]))  
        ax2.set_ylim(min(p1[:,1]), max(p1[:,1]))  
        ax2.set_zlim(min(p1[:,2]), max(p1[:,2]))   

        ax1.set_xlim(min(p1[:,0]), max(p1[:,0]))  
        ax1.set_ylim(min(p1[:,1]), max(p1[:,1]))  
        ax1.set_zlim(min(p1[:,2]), max(p1[:,2]))           
      
        lc = Line3DCollection(ls, linewidths=0.3, colors=clrs, alpha=0.5)
        ax2.add_collection(lc)

        plt.savefig('./'+str(itr)+'_event.pdf') 

        ax2.set_xlim(min(z), max(z))  
        ax2.set_ylim(min(y), max(y))  
        ax2.set_zlim(min(x), max(x))   

        ax1.set_xlim(min(z), max(z))  
        ax1.set_ylim(min(y), max(y))  
        ax1.set_zlim(min(x), max(x))         

        plt.savefig('./'+str(itr)+'_event_zoom.pdf') 
        
        # pdb.set_trace()