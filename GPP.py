#!/usr/bin/env python
# coding: utf-8

# In[1]:


cd ../track_gpp


##################################################################################################
##################################################################################################
############ calculation of track of cyclone and tracing in on the base projection ###############
##################################################################################################
##################################################################################################
############################### file format is xlsx and csv ######################################

# In[2]:


import numpy as np
from colour import Color
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.collections as mcoll
import matplotlib.path as mpath
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from colorspacious import cspace_converter
from collections import OrderedDict
from mpl_toolkits.basemap import Basemap
import  matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

cmaps = OrderedDict()


# In[3]:


df = pd.read_excel("track_data.xlsx",sheet_name='dat')
dF = pd.read_excel("track_data.xlsx",sheet_name='dat1')
lat1 = df.lat.values
lon1 = df.lon.values
gpp1 = df.gpp.values
idd1 = dF.id.values
lat = lat1[:]
lon = lon1[:]
gpp = gpp1[:]
idd = idd1[:]
tck =np.loadtxt('track.txt',dtype=float)


# In[4]:


lat = lat.tolist()
lon = lon.tolist()
gpp = gpp.tolist()
idd = idd.tolist()


# In[5]:


gpp


# KT = []
# tck_sum = 0
# for i in range(len(tck)):
#     tck_sum += tck[i]
#     KT.append(tck_sum)

# np.savetxt("Data.txt",KT)

# for i in range(len(idd)+1):    
#     if len(lat[int(idd[i]):int(idd[i+1])]) == tck[i]:
#         print("True")
#     else:
#         print("=========")

# In[107]:


fig = plt.figure(figsize=(15,20))
#fig,ax =plt.subplots()
#ax = fig.add_subplot()
m = Basemap(projection='cyl', llcrnrlat=0, urcrnrlat=30,llcrnrlon=46, urcrnrlon=82, resolution='c', area_thresh=1005.)
m.drawstates()
m.bluemarble()
m.drawcoastlines(linewidth=1.0)
m.drawcountries(linewidth=1.0)
m.drawstates(linewidth=1.0)
m.drawparallels(np.arange(0,30,5.),labels=[1,0,0,1])
m.drawmeridians(np.arange(46,82,5.),labels=[1,0,0,1])
m.drawmapboundary(fill_color='aqua')
for i in range(len(idd)-1):
    latitude = lat[int(idd[i]):int(idd[i+1])]
    longitude = lon[int(idd[i]):int(idd[i+1])]
    gpp12 = gpp[int(idd[i]):int(idd[i+1])]
    x,y = m(longitude, latitude)
    #m.colorbar(location='right')
    plt.plot(x,y,'.',color ="white")
    def colorline(x, y, z=gpp12, cmap=plt.get_cmap('jet'), norm=plt.Normalize(0.0,1.0),
        linewidth=3, alpha=1.0):
        if z is gpp12:
            z = np.linspace(0.0, 1.0, len(x))
        if not hasattr(z, "__iter__"):
            z = np.array([z])
        z = np.asarray(z)
        segments = make_segments(x, y)
        lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,linewidth=linewidth, alpha=alpha)
        ax = plt.gca()
        ax.add_collection(lc)
        return lc

    def make_segments(x, y):
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        return segments
    
    path = mpath.Path(np.column_stack([x, y]))
    verts = path.interpolated(steps=3).vertices
    x, y = verts[:, 0], verts[:, 1]
    z = np.linspace(0, 1, len(x))
    plot = colorline(x,y,gpp12, cmap=plt.get_cmap('jet'), linewidth=3)
    #sc = plt.scatter(x, y,c='white',s=0,vmin=0,vmax=30, cmap=plt.cm.jet)
#fig.colorbar(plot,shrink=0.6,extend='both',orientation='vertical')
cbar = fig.colorbar(plot, ticks=[-1, 0, 1],fraction=0.039,pad=0.02)
cbar.set_label('Genesis potential parameter',size=20, rotation=270)
cbar =cbar.ax.set_yticklabels(['< 0', '10', '> 30'])
rcParams['axes.titlepad'] = 50 
#plt.xlabel('Longitude')
#plt.ylabel('latitude')
plt.title('Genesis Potential parameter variation over Arabian sea cyclone tracks(1982-2018)',size='18')
plt.savefig("Sample.tiff")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


`


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




