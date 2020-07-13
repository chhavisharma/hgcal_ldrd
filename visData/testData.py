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
    
    data = np.load('./../../data/raw/partGun_PDGid15_x1000_Pt3.0To100.0_NTUP_2_hgcal_graph_pos_evt71.npz')
    
    print(data)
    keys = [k for k in data.keys()]

    print('Keys',keys)
    for k in keys:
        print(k,':',data[k].shape)

    #X = data[keys[0]]
    
    ''' Read Files '''
    events = []
    root = './../../data/raw/'
    for file in os.listdir(root):
        if file.endswith(".npz"):
            path = os.path.join(root, file)
            events.append(path)

    for itr,event_path in enumerate(events):
        
        print('event:',itr)

        item  = np.load(event_path)
        
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
        
        spRi, spRo, ytruth = sparse_to_graph(x, Ri_rows, Ri_cols, Ro_rows, Ro_cols, ytruth)

        pdb.set_trace()

        point3Din = np.concatenate((z[Ri_rows,np.newaxis,np.newaxis],y[Ri_rows,np.newaxis,np.newaxis],x[Ri_rows,np.newaxis,np.newaxis]), axis=2)
        point3Dout = np.concatenate((z[Ro_rows,np.newaxis,np.newaxis],y[Ro_rows,np.newaxis,np.newaxis],x[Ro_rows,np.newaxis,np.newaxis]), axis=2)
        segments = np.concatenate((point3Din, point3Dout), axis=1)

        pdb.set_trace()

        '''
        # Creating figure 
        fig = plt.figure(figsize = (20, 9)) 
        ax1 = fig.add_subplot(121,projection='3d')
        ax2 = fig.add_subplot(122,projection='3d')
                
        #Axis 1 - hits 
        ax1.set_xlabel('Z-axis', fontweight ='bold')  
        ax1.set_ylabel('Y-axis', fontweight ='bold')  
        ax1.set_zlabel('X-axis', fontweight ='bold') 
        ax1.scatter3D(z, y, x, s=1, color = "green"); 

        # Axis 2 - Edges
        ax2.set_xlabel('Z-axis', fontweight ='bold')  
        ax2.set_ylabel('Y-axis', fontweight ='bold')  
        ax2.set_zlabel('X-axis', fontweight ='bold')
        # Create the 3D-line collection object
        print('segments' ,segments.shape)
        segments = segments[:10]
        for i in range(segments.shape[0]):
            ax2.plot([segments[i,0,0], segments[i,1,0]], [segments[i,0,1], segments[i,1,1]],zs=[segments[i,0,2], segments[i,1,2]], linewidth=1,color="red")

        # show plot 
        plt.show()
        # plt.savefig('./'+str(i)+'_event_sample.pdf') 
        
        '''

        if(itr==2):
            break
        
        #plot
        depth = x
        print(segments[0])
        segments = np.array([[[1,1,1],[10,10,10]],[[1,2,3],[6,7,8]]])
        lc = Line3DCollection(segments[:10], color='green' )#,norm=plt.Normalize(x.min(),x.max()) )
        lc.set_linewidth(1)
        # lc.set_array(x) 
        fig = plt.figure()
        
        ax = fig.gca(projection='3d')
        ax.set_zlim(min(x), max(x))
        ax.set_ylim(min(y), max(y))
        ax.set_xlim(min(z), max(z))

        print('depth limits = ', min(x), max(x))
        plt.title('3D-Figure-'+str(itr))
        ax.add_collection3d(lc, zs=x, zdir='z')
        plt.show()


