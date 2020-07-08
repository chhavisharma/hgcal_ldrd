import os
import numpy as np
import scipy as scp
import torch

from  mpl_toolkits import mplot3d
import matplotlib.pyplot as plt 

import pdb

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
    root = '/home/csharma/workspace/data/raw/'
    for file in os.listdir(root):
        if file.endswith(".npz"):
            path = os.path.join(root, file)
            events.append(path)

    for i,event_path in enumerate(events):
        
        print('event:',i)

        item  = np.load(event_path)
        pdb.set_trace()
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

        # Creating figure 
        fig = plt.figure(figsize = (12, 9)) 
        ax = plt.axes(projection ="3d") 

        ax.set_xlabel('Z-axis', fontweight ='bold')  
        ax.set_ylabel('Y-axis', fontweight ='bold')  
        ax.set_zlabel('X-axis', fontweight ='bold') 
      
        # Creating plot 
        ax.scatter3D(z, y, x, s=1, color = "green"); 
        plt.title("single event plot | 25 us") 
        
        # show plot 
        plt.savefig('./'+str(i)+'_event.pdf') 

        #pdb.set_trace() 
        if(i==9):
            break


