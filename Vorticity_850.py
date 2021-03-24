#!/usr/bin/env python
# coding: utf-8

# In[5]:

cd ../VORTICITY/RV_850/

##################################################################################################
##################################################################################################
############### calculation of vorticity from the gridded data of ERA5 and avhrr #################
##################################################################################################
##################################################################################################
##################################### file format is netCDF ######################################

# In[6]:


import csv
from scipy.io import netcdf
from cdo import*
import random
import math
from numpy import nan
from mpl_toolkits.basemap import Basemap
import xarray as xr
import pandas as pd
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pylab import rcParams
from netCDF4 import Dataset
from sklearn import datasets, linear_model, metrics
cdo = Cdo()                                            #python interface for cdo to work
cdo.debug = True


# In[7]:


from mpl_toolkits.basemap import Basemap
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
import matplotlib.patches as mpatches

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

cmaps = OrderedDict()


df = pd.read_excel("diverg.xlsx",sheet_name='Sheet1')
Lat1 = df.lat.values
Lon1 = df.lon.values
date = df.date.values
cat = df.category.values
year = df.year.values
lat1 = Lat1 - 0.15
lat2 = Lat1 + 0.15
lon1 = Lon1 - 0.15
lon2 = Lon1 + 0.15


D = []
os.remove('VO850.txt')
for i in range(len(date)):
    filename = str(date[i])+'.nc'
    Latt1 = lat1[i]
    Latt2 = lat2[i]
    Long1 = lon1[i]
    Long2 = lon2[i]
    cdo.sellonlatbox(Long1,Long2,Latt1,Latt2, input = filename , output = str(date[i])+str(Long2)+'.nc' , options = '-f nc')
    filename = str(date[i])+str(Long2)+'.nc'
    dataset = nc.Dataset(filename)
    dataset.variables.keys()
    dataset.dimensions.keys()
    open_netcdf = xr.open_dataset(filename)
    dataset = open_netcdf.to_dataframe()
    dd = dataset.vo.values    
    os.remove(str(date[i])+str(Long2)+'.nc')
    SS = []
    TT = []
    for i in range(len(dd)):
        if dd[i] == nan:
            SS.append(dd[i])   #This is a dummy variable and SS remain as [] and nothing happens as nan wont be appended.
        else:
            TT.append(dd[i])   #This is the most needed one and every values will be appended to this list.
    TT = [tt for tt in TT if str(tt) != 'nan']
    averd = np.average(TT) #averagind the value sin the grid
    #D.append(averd)    #appending the grid averaged values to X1 list
    #x1 = np.asarray(D) #appending X1 asarray in x1
    #div = np.max(D)
    f = open("VO850.txt","a+")
    f.write("%s\r\n" % averd)   # %s is used for the decimal points and %d is used for the ones tenths sand other places
    f.close()   


# In[10]:


S=[]
for i in range(len(date)):
        S.append(i)


# In[11]:


VO850 = np.loadtxt("VO850.txt")
Z = np.polyfit(S,VO850,1)


# In[12]:


Z_Vo = []
for i in range(len(S)):
    Zv = Z[0]*S[i]+Z[1]
    Z_Vo.append(Zv)


# In[13]:


Fig = plt.figure(figsize=(15,6),facecolor='gainsboro')
ax = plt.axes()
ax.set_facecolor("lightyellow")
plt.rcParams['axes.xmargin'] = 0.005
plt.rcParams['axes.ymargin'] = 0.
plt.xlabel("Cyclones",fontsize='20',weight='bold')
plt.ylabel("Vorticity ($s^{-1}$)",fontsize='20',weight='bold')
plt.xticks(fontsize='15',weight='bold')
plt.yticks(fontsize='15',weight='bold')
plt.ylim(-0.00028,0.0016)
ax.grid(zorder=0)

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
plt.bar(S,VO850,color='lightseagreen',label = 'Vorticity')
plt.plot(S,Z_Vo,color='black',linestyle=(0,(1,1)),label='trend = 7.53x10$^{-6}$sec$^{-1}$')
ax.legend()

for i in range(len(S)):
    plt.annotate(str(cat[i]), 
                 xy=(S[i]+0.05,VO850[i]+0.000018), #definfing the place 
                 #where you need the annotations
                 va="bottom",color='maroon',
                 ha="center",fontsize='15',rotation='90',weight='bold')
    plt.annotate(str(year[i]), 
                 xy=(S[i]+0.05,0),
                 va="top",color='brown',
                 ha="center",fontsize='15',rotation='90',weight='bold')

plt.title("Variation of vorticity at 850hPa per cyclone on the day of maximum sustained wind speed",weight='bold',fontsize='15')
plt.savefig("VO850.jpeg")


