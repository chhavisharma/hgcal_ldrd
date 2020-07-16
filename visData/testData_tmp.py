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


        # get nodeid Edge pairs
        incoming = [[z[n], y[n], x[n], n,e,c] for n, e, c in zip (Ri_rows, Ri_cols, ytruth)]
        outgoing = [[z[n], y[n], x[n], n,e,c] for n, e, c in zip (Ro_rows, Ro_cols, ytruth)]

        # sort by edge id
        sin = sorted(incoming, key=lambda x: x[4])
        sot = sorted(outgoing,  key=lambda x: x[4])

        # Creating figure 
        fig = plt.figure(figsize = (20, 15)) 
        ax1 = fig.add_subplot(221,projection='3d')   
        ax2 = fig.add_subplot(222,projection='3d') 
        ax3 = fig.add_subplot(223,projection='3d') 
        ax4 = fig.add_subplot(224,projection='3d') 
        
        ax1.title.set_text('All')
        ax2.title.set_text('Hadron')
        ax3.title.set_text('EM')
        ax4.title.set_text('MIP')
        
        ax1.set_xlabel('Z-axis', fontweight ='bold')  
        ax1.set_ylabel('Y-axis', fontweight ='bold')  
        ax1.set_zlabel('X-axis', fontweight ='bold')  
        
        ax2.set_xlabel('Z-axis', fontweight ='bold')  
        ax2.set_ylabel('Y-axis', fontweight ='bold')  
        ax2.set_zlabel('X-axis', fontweight ='bold')   

        ax3.set_xlabel('Z-axis', fontweight ='bold')  
        ax3.set_ylabel('Y-axis', fontweight ='bold')  
        ax3.set_zlabel('X-axis', fontweight ='bold')  
        
        ax4.set_xlabel('Z-axis', fontweight ='bold')  
        ax4.set_ylabel('Y-axis', fontweight ='bold')  
        ax4.set_zlabel('X-axis', fontweight ='bold') 

        p1s = np.array(sin[:])[:,:3]
        p2s = np.array(sot[:])[:,:3]
        cs = np.array(sin[:])[:,5]

        nonoise = cs!=0.0
        HDMask = cs==1.0
        EMMask = cs==2.0
        MPMask = cs==3.0

        cdict = {0:'grey', 1:'red', 2:'blue', 3:'green'}
        

        #Axis 1 - all hits 
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
        ax3.set_xlim(min(p1[:,0]), max(p1[:,0]))  
        ax3.set_ylim(min(p1[:,1]), max(p1[:,1]))  
        ax3.set_zlim(min(p1[:,2]), max(p1[:,2])) 
        ax4.set_xlim(min(p1[:,0]), max(p1[:,0]))  
        ax4.set_ylim(min(p1[:,1]), max(p1[:,1]))  
        ax4.set_zlim(min(p1[:,2]), max(p1[:,2]))                         

        clrs = [ cdict[int(x)] for x in cs[nonoise]]
        lc = Line3DCollection(ls, linewidths=0.3, colors=clrs, alpha=0.5)
        ax1.add_collection(lc)
        
        if(sum(HDMask)>0):
            # Axis 2 - Hadrons
            p1 = p1s[HDMask] 
            p2 = p2s[HDMask] 
            ls = np.hstack([p1,p2]).copy()  
            ls = ls.reshape((-1,2,3))  

            # ax2.set_xlim(min(p1[:,0]), max(p1[:,0]))  
            # ax2.set_ylim(min(p1[:,1]), max(p1[:,1]))  
            # ax2.set_zlim(min(p1[:,2]), max(p1[:,2]))           
        
            clrs = [ cdict[int(x)] for x in cs[HDMask]]
            lc = Line3DCollection(ls, linewidths=0.3, colors=clrs, alpha=0.5)
            ax2.add_collection(lc)

        if(sum(EMMask)>0):
            # Axis 2 - EM
            p1 = p1s[EMMask] 
            p2 = p2s[EMMask] 
            ls = np.hstack([p1,p2]).copy()  
            ls = ls.reshape((-1,2,3))  

            # ax3.set_xlim(min(p1[:,0]), max(p1[:,0]))  
            # ax3.set_ylim(min(p1[:,1]), max(p1[:,1]))  
            # ax3.set_zlim(min(p1[:,2]), max(p1[:,2]))           
        
            clrs = [ cdict[int(x)] for x in cs[EMMask]]
            lc = Line3DCollection(ls, linewidths=0.3, colors=clrs, alpha=0.5)
            ax3.add_collection(lc)

        if(sum(MPMask)>0):
            # Axis 3 - MIPS
            p1 = p1s[MPMask] 
            p2 = p2s[MPMask] 
            ls = np.hstack([p1,p2]).copy()  
            ls = ls.reshape((-1,2,3))  

            # ax4.set_xlim(min(p1[:,0]), max(p1[:,0]))  
            # ax4.set_ylim(min(p1[:,1]), max(p1[:,1]))  
            # ax4.set_zlim(min(p1[:,2]), max(p1[:,2]))           
        
            clrs = [ cdict[int(x)] for x in cs[MPMask]]
            lc = Line3DCollection(ls, linewidths=0.3, colors=clrs, alpha=0.5)
            ax4.add_collection(lc)

        plt.savefig('./split/'+str(itr)+'_event.pdf') 
        
        # pdb.set_trace()