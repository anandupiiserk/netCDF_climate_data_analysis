#!/usr/bin/env python
# coding: utf-8

# In[1]:
##################################################################################################
##################################################################################################
########### calculation of Central pressure value from the IMD dataset and JTWC datas ############
##################################################################################################
##################################################################################################
############################## file format is netCDF,xlsx and csv ################################

cd ../CP/old_data_fig/


# In[2]:


import csv
import cv2
from scipy.io import netcdf
from itertools import islice 
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
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

cmaps = OrderedDict()


# In[3]:


ls


# In[5]:


dt = pd.read_excel('CP_variation.xlsx',sheetname = 'Sheet1')
df = pd.read_excel('CP_variation.xlsx',sheetname = 'Sheet2')
dg = pd.read_excel('CP_variation.xlsx',sheetname = 'Sheet3')
freq = df.freq.values #frequency of occurance of cyclones
cat = dg.category.values
time = dt.time.values #time of of duratio occurance of cyclone
CP = dt.CP.values       #Maximum sustained wind speed in knots
terms = dg.terms.values #number of intervals of each duration
year = dg.year.values   #year of occurance of cyclones


# # creating a 43 elemented array(43 cyclones)

# fre = [0, 10,  13,  17,  26,  41,  51,  70,  83,  87,  97, 117, 120, 138,164, 215, 230, 236, 245, 259, 297, 311, 326, 344, 383, 385, 391,399, 434, 439, 445, 464, 500, 529, 575, 613, 633, 667, 718, 769,786, 839, 899, 909]

# In[6]:


SUM = []                         #we are slicing the values to each cyclonic duration into 43 arrays
for i in range(0,len(freq)-1):
    S = time[freq[i]:freq[i+1]]
    SUM.append(S)


# In[7]:


TTT = []   #dummy variable
for i in range(len(terms)):
    KH = SUM[i]                  #dummy variable
    for j in range(len(KH)-1):
        diff = KH[j+1]-KH[j]     #calculating the difference in the time in hours
        TTT.append(diff)         


# In[8]:


for n,i in enumerate(TTT): #there are non uniform intervals hence 0000-1200 = "-12" similarly corrected others
    if i == -1200:
        TTT[n] = 1200
    if i == -900:
        TTT[n] = 1500
    if i == -1800:
        TTT[n] = 600
    if i == -2100:
        TTT[n] = 300


# In[9]:


TERMS = terms-1
kkk = 0     #dummy variable
fre = []    #dummy variable
for i in range(len(TERMS)):
    kkk += TERMS[i]                #running average to make the intervals uniform
    fre.append(kkk)
fre.insert(0,0)       #inserting zero in the first index


# now fre and TTT are the required parameters and we need to calculate the duration of each cyclone

# In[10]:


ght = []
ggg = 0
for i in range(len(fre)-1):
    ggg = TTT[fre[i]:fre[i+1]]  #slicing the TTT duration to arrays with -1 values from each
    ght.append(ggg)


# In[11]:


get_ipython().run_line_magic('pinfo', 'ght')


# In[12]:


ggg = 0
Y = []
for i in range(len(ght)):
    ggg = np.sum(ght[i])
    Y.append(ggg)


# In[13]:


Y


# In[14]:


y = []
for i in range(len(Y)):
    z = Y[i]/100
    y.append(z)


# In[15]:


S=[]
for i in range(len(y)):
        S.append(i)


# In[16]:


z = np.polyfit(S,y,1)


# In[17]:


Z = []
for i in range(len(S)):
    dg = z[0]*S[i]+z[1]
    Z.append(dg)


# In[18]:


z


# In[19]:


labels = ["C%d" % i for i in range(len(S))]
# Setting the background color
Fig = plt.figure(figsize=(20,7),facecolor='gainsboro')
plt.rcParams['axes.xmargin'] = 0.005
plt.rcParams['axes.ymargin'] = 0.
ax = plt.axes()
ax.grid(zorder=0)
plt.rc('grid', linestyle="-", color='black')
ax.set_facecolor("seashell")
plt.rc('grid', linestyle="-", color='black')

plt.xlabel("Cyclones",fontsize='20',weight='bold')
plt.ylabel("Duration (in hrs)",fontsize='20',weight='bold')

plt.bar(S,y,color='sienna',label='duration',zorder=2)
plt.plot(S,Z,color='black',linestyle='dashed',label='trend = 1.69hr/cyclone')

#plt.bar(S,Min_val,color='lightcoral',label='Convergence',width=0.67)
#plt.plot(S,Zmn,color='midnightblue',linestyle='dotted')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 1
plt.rcParams['grid.linestyle'] = "dotted"
plt.rcParams['grid.color'] = "grey"
plt.yticks(np.arange(0, 240, 24.0),fontsize='15',weight='bold')
plt.xticks(fontsize='15',weight='bold')
plt.ylim(-28,216)
ax.legend(loc='upper left',fontsize='15')
for i in range(len(S)):
    plt.annotate(str(year[i]), 
                 xy=(S[i],0),
                 va="top",
                 ha="center",fontsize='14',rotation='90',weight='bold')
    plt.annotate(str(cat[i]), 
                 xy=(S[i]+0.05,y[i]), #definfing the place where you need the annotations
                 va="bottom",color='black',
                 ha="center",fontsize='14',rotation='90',weight='bold')    
plt.title("Duration of cyclone",fontsize='20',weight='bold')
plt.savefig("duration_of_cyclones.jpeg")


# array([ 1.54258532, 35.62896406])

# In[20]:


HH = []                     #Slicing to 43 arrays
for i in range(len(terms)):
    sh = CP[freq[i]:freq[i+1]] #taking one element only
    HH.append(sh)


# In[25]:


type(freq)


# In[21]:


CP_s = []
for i in range(len(HH)):
    av = np.average(HH[i])
    CP_s.append(av)


# In[22]:


z = np.polyfit(S,CP_s,1)
Z = []
for i in range(len(S)):
    dg = z[0]*S[i]+z[1]
    Z.append(dg)


# labels = ["C%d" % i for i in range(len(S))]
# # Setting the background color
# Fig = plt.figure(figsize=(20,7),facecolor='gainsboro')
# ax = plt.axes()
# ax.set_facecolor("seashell")
# 
# plt.xlabel("Cyclones",fontsize='20',weight='bold')
# plt.ylabel("Central pressure ()",fontsize='20',weight='bold')
# 
# plt.bar(S,CP_s,color='dodgerblue',label='Central Pressure (mbar)')
# plt.plot(S,Z,color='black',linestyle='dashed',label='trend')
# 
# #plt.bar(S,Min_val,color='lightcoral',label='Convergence',width=0.67)
# #plt.plot(S,Zmn,color='midnightblue',linestyle='dotted')
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False)
# 
# plt.yticks(fontsize='15',weight='bold')
# plt.yticks(np.arange(0, 2000, 500),fontsize='15',weight='bold')
# plt.xticks(fontsize='15',weight='bold')
# plt.ylim(-250)
# ax.legend()
# for i in range(len(S)):
#     plt.annotate(str(cat[i]), 
#                  xy=(S[i],CP_s[i]),
#                  va="bottom",
#                  ha="center",fontsize='14',rotation='90',weight='bold')
#     plt.annotate(str(year[i]), 
#                  xy=(S[i],0),
#                  va="top",
#                  ha="center",fontsize='14',rotation='90',weight='bold')
#     
# plt.savefig("CP.jpeg")

# array([1.05512985, 4.25819768])

# # The below code works

# fig, ax1 = plt.subplots(figsize=(25,8),facecolor='gainsboro')
# ax1.set_facecolor("seashell")
# ax.yaxis.grid()
# ax.grid(zorder=0)
# plt.margins(0)
# plt.yticks(fontsize='20',weight='bold')
# plt.xticks(fontsize='15',weight='bold')
# ax1.bar(S, CP_s,color = 'blue',label='Central pressure',zorder=3)
# #plt.ylim(-200)
# plt.rcParams['axes.grid'] = False
# plt.yticks(np.arange(900, 1050, 50),fontsize='15',weight='bold')
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False)
# for i in range(len(S)):
#     plt.annotate(str(cat[i]), 
#                  xy=(S[i],CP_s[i]),
#                  va="bottom",
#                  ha="center",fontsize='14',rotation='90',weight='bold')
#     plt.annotate(str(year[i]), 
#                  xy=(S[i],0),
#                  va="top",
#                  ha="center",fontsize='14',rotation='90',weight='bold')
# 
# ax2 = ax1.twinx()
# ax2.plot(S, Z,'k--',label = 'trend')
# plt.rcParams['axes.grid'] = True
# #fig.legend(loc='upper left',fontsize='15')
# fig.legend(loc="upper right", bbox_to_anchor=(1.28,1.145), bbox_transform=ax.transAxes,fontsize='20')
# 
# #plt.yticks(np.arange(min(Z)-4.7102835928417, max(Z)+10.7102835928417, 6),fontsize='15',weight='bold')
# ax1.set_xlabel('Cyclones',color='black',weight='bold',fontsize='20')
# ax2.get_yaxis().set_visible(True)
# ax1.set_ylabel('Central pressure (mbar)', color='black',weight='bold',fontsize='20')
# ax2.set_ylabel('Central pressure trend', color='black',weight='bold',fontsize='20')
# plt.yticks(np.arange(900, 1050, 50),fontsize='15',weight='bold')
# plt.title("Central pressure variation along cyclonic tracks",weight='bold',fontsize='20')  
# plt.savefig("CP.jpeg")
# plt.ylim(900,1000)
# plt.show()
array([-2.69599998e-01,  9.93494284e+02])
# In[ ]:


# In[23]:


fig, ax = plt.subplots(figsize=(26,8),facecolor='gainsboro')
ax.grid(zorder=0)
ax.set_facecolor("seashell")
plt.yticks(fontsize='15',weight='bold')
plt.xticks(fontsize='15',weight='bold')
plt.ylim(940,1010)
plt.ylabel("Central pressure (hpa)",fontsize='15',weight='bold')
plt.xlabel("Cyclone",labelpad=50,fontsize='15',weight='bold')
plt.plot(S,Z,'k--',label = 'trend = -2.7hpa/cyclone')
plt.bar(S,CP_s,color='purple',zorder=2,label = 'Central pressure')
plt.legend(fontsize='15')
for i in range(len(S)):
    plt.annotate(str(cat[i]), 
                 xy=(S[i],CP_s[i]),
                 va="bottom",
                 ha="center",fontsize='14',rotation='90',weight='bold')
    plt.annotate(str(year[i]), 
                 xy=(S[i],940),
                 va="top",
                 ha="center",fontsize='14',rotation='90',weight='bold')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
#fig.legend(loc="upper right", bbox_to_anchor=(1.28,1.145), bbox_transform=ax.transAxes,fontsize='20')
#fig.legend()
plt.title("Averaged Central pressure variation along cyclonic tracks",weight='bold',fontsize='20')  
plt.savefig("Pressure.jpg")




