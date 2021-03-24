#!/usr/bin/env python
# coding: utf-8

# In[1]:

##################################################################################################
##################################################################################################
##### calculation of speed and total distance travelled by cyclone fromt he avaliable datasets ###
##################################################################################################
##################################################################################################
##################################### file format is netCDF,xlsx #################################

cd ../path_speed


# In[3]:


import csv
import cv2
from math import radians, cos, sin, asin, sqrt 
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


# In[4]:


dt = pd.read_excel('data_ACE_calculation.xlsx',sheetname = 'Sheet1')
df = pd.read_excel('data_ACE_calculation.xlsx',sheetname = 'Sheet2')
dg = pd.read_excel('data_ACE_calculation.xlsx',sheetname = 'Sheet3')
freq = df.freq.values #frequency of occurance of cyclones
cat = dg.category.values
lat = dt.lat.values
lon = dt.lon.values
time = dt.time.values #time of of duratio occurance of cyclone
#V = dt.V.values       #Maximum sustained wind speed in knots
terms = dg.terms.values #number of intervals of each duration
year = dg.year.values   #year of occurance of cyclones


# # creating a 46 elemented array(46 cyclones)

# fre = [0, 10,  13,  17,  26,  41,  51,  70,  83,  87,  97, 117, 120, 138,164, 215, 230, 236, 245, 259, 297, 311, 326, 344, 383, 385, 391,399, 434, 439, 445, 464, 500, 529, 575, 613, 633, 667, 718, 769,786, 839, 899, 909]

# In[6]:


sl_lat = []                         #we are slicing the values to each cyclonic duration into 43 arrays
for i in range(0,len(freq)-1):
    S = lat[freq[i]:freq[i+1]]
    sl_lat.append(S)


# In[7]:


sl_lon = []                         #we are slicing the values to each cyclonic duration into 43 arrays
for i in range(0,len(freq)-1):
    S = lon[freq[i]:freq[i+1]]
    sl_lon.append(S)


# In[24]:


dist = []
for i in range(len(sl_lat)):
    dumm = []
    for j in range((len(sl_lat[i])-1)):
        lat1 = radians(sl_lat[i][j])
        lat2 = radians(sl_lat[i][j+1])
        lon1 = radians(sl_lon[i][j])
        lon2 = radians(sl_lon[i][j+1])
        #dumm = []
        #def distance(lat1, lat2, lon1, lon2): 
            #lon1 = radians(lon1) 
            #lon2 = radians(lon2) 
            #lat1 = radians(lat1) 
            #lat2 = radians(lat2) 
        # Haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * asin(sqrt(a)) 
        # Radius of earth in kilometers. Use 3956 for miles 
        r = 6371
        # calculate the result 
        dumm.append(c*r)
        #return(c * r)
    dist.append(dumm)
        
# driver code 
#print(distance(lat1, lat2, lon1, lon2),"km") 


# In[25]:


TERMS = terms-1
kkk = 0     #dummy variable
fre = []    #dummy variable
for i in range(len(TERMS)):
    kkk += TERMS[i]                #running average to make the intervals uniform
    fre.append(kkk)
fre.insert(0,0)       #inserting zero in the first index


# In[26]:


distance = []
for i in range(len(dist)):
    av = np.sum(dist[i])
    distance.append(av)


# In[27]:


S=[]
for i in range(len(distance)):
        S.append(i)


# In[28]:


z = np.polyfit(S,distance,1)
Z = []
for i in range(len(S)):
    dg = z[0]*S[i]+z[1]
    Z.append(dg)


# In[29]:


z


# In[40]:


fig, ax1= plt.subplots(figsize=(25,8),facecolor='gainsboro')
plt.rcParams['axes.grid'] = False
ax.grid(zorder=0)
ax1.set_facecolor("seashell")
plt.yticks(fontsize='20',weight='bold')
plt.xticks(fontsize='15',weight='bold')
ax1.bar(S, distance,color = 'lightslategray',label='track speed',zorder=3)
#plt.ylim(-250,2500)
plt.rcParams['axes.xmargin'] = 0.005
plt.rcParams['axes.ymargin'] = 0.
plt.yticks(np.arange(-300,3000,500),fontsize='15',weight='bold')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)


for i in range(len(S)):
    plt.annotate(str(cat[i]), 
                 xy=(S[i],distance[i]),
                 va="bottom",
                 ha="center",fontsize='14',rotation='90',weight='bold')
    plt.annotate(str(year[i]), 
                 xy=(S[i],0),
                 va="top",
                 ha="center",fontsize='14',rotation='90',weight='bold')
ax2 = ax1.twinx()
ax2.plot(S, Z,'k--',label = 'trend = 11.33km/cyclone')
plt.rcParams['axes.grid'] = True
#fig.legend(loc='upper left',fontsize='15')
fig.legend(loc="upper right", bbox_to_anchor=(0.268,0.99), bbox_transform=ax.transAxes,fontsize='20')
#fig.legend()
plt.ylim(min(Z)-100,max(Z)+100)
plt.yticks(fontsize='15',weight='bold')
ax1.set_xlabel('Cyclonic years',color='black',weight='bold',fontsize='20')
ax2.get_yaxis().set_visible(True)
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelright=False)
ax1.set_ylabel('Distance travelled (km)', color='black',weight='bold',fontsize='20')
#ax2.set_ylabel('trend variation', color='black',weight='bold',fontsize='20')
plt.title("Path length travelled per cyclone variation from 1982 to 2019",weight='bold',fontsize='20')  
plt.savefig("path_length.jpeg")
plt.show()


# In[36]:


pwd


# # speed of propogation of cyclone

# In[14]:


SUM = []                         #we are slicing the values to each cyclonic duration into 43 arrays
for i in range(0,len(freq)-1):
    S = time[freq[i]:freq[i+1]]
    SUM.append(S)
    
TTT = []   #dummy variable
for i in range(len(terms)):
    KH = SUM[i]                  #dummy variable
    for j in range(len(KH)-1):
        diff = KH[j+1]-KH[j]     #calculating the difference in the time in hours
        TTT.append(diff)         

for n,i in enumerate(TTT): #there are non uniform intervals hence 0000-1200 = "-12" similarly corrected others
    if i == -1200:
        TTT[n] = 1200
    if i == -900:
        TTT[n] = 1500
    if i == -1800:
        TTT[n] = 600
    if i == -2100:
        TTT[n] = 300

TERMS = terms-1
kkk = 0     #dummy variable
fre = []    #dummy variable
for i in range(len(TERMS)):
    kkk += TERMS[i]                #running average to make the intervals uniform
    fre.append(kkk)
fre.insert(0,0)       #inserting zero in the first index

ght = []
ggg = 0
for i in range(len(fre)-1):
    ggg = TTT[fre[i]:fre[i+1]]  #slicing the TTT duration to arrays with -1 values from each
    ght.append(ggg)

ggg = 0
Y = []
for i in range(len(ght)):
    ggg = np.sum(ght[i])
    Y.append(ggg)

y = []
for i in range(len(Y)):
    z = Y[i]/100
    y.append(z)


# ght and dist are the required to calculated the speed

# time = []
# for i in range(len(time)):
#     tt = []
#     for j in range(len(time[i])-1):
#         z = (ght[i][j])/100
#         tt.append(z)
#     time.append(tt)

# In[15]:


ppp = []
for i in range(len(dist)):
    Speed = []
    for j in range(len(dist[i])):
        SPEED = (dist[i][j]/ght[i][j])*100
        Speed.append(SPEED)
    ppp.append(Speed)


# In[16]:


speed = []
for i in range(len(ppp)):
    av = np.average(ppp[i])
    speed.append(av)


# In[17]:


S=[]
for i in range(len(distance)):
        S.append(i)


# In[18]:


z = np.polyfit(S,speed,1)
Z = []
for i in range(len(S)):
    dg = z[0]*S[i]+z[1]
    Z.append(dg)


# In[19]:


z


# In[20]:


fig, ax1 = plt.subplots(figsize=(25,8),facecolor='gainsboro')
plt.rcParams['axes.grid'] = False
ax1.grid(zorder=0)
ax1.set_facecolor("seashell")
plt.yticks(fontsize='20',weight='bold')
plt.xticks(fontsize='15',weight='bold')
ax1.bar(S, speed,color = 'darkred',label='track speed',zorder=3)
plt.ylim(-5)
plt.rcParams['axes.xmargin'] = 0.005
plt.rcParams['axes.ymargin'] = 0.
plt.yticks(np.arange(0, 54, 8),fontsize='15',weight='bold')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
for i in range(len(S)):
    plt.annotate(str(cat[i]), 
                 xy=(S[i],speed[i]),
                 va="bottom",
                 ha="center",fontsize='14',rotation='90',weight='bold')
    plt.annotate(str(year[i]), 
                 xy=(S[i],0),
                 va="top",
                 ha="center",fontsize='14',rotation='90',weight='bold')
ax2 = ax1.twinx()
ax2.plot(S, Z,'k--',label = 'trend = -0.12km/hr')
plt.rcParams['axes.grid'] = True
#fig.legend(loc='upper left',fontsize='15')
fig.legend(loc="upper right", bbox_to_anchor=(4.69,2.2), bbox_transform=ax.transAxes,fontsize='20')
#plt.rcParams['axes.xmargin'] = 0
#plt.rcParams['axes.ymargin'] = 0
plt.yticks(np.arange(min(Z)-2.621150547548936, max(Z)+9.695571063081193,5),fontsize='15',weight='bold')
ax1.set_xlabel('Cyclonic years',color='black',weight='bold',fontsize='20')
ax2.get_yaxis().set_visible(True)
ax1.set_ylabel('track speed (km/hr)', color='black',weight='bold',fontsize='20')
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelright=False)
#ax2.set_ylabel('velocity trend', color='black',weight='bold',fontsize='20')
plt.title("Track speed of cyclones(in km/hr)",weight='bold',fontsize='20')  
plt.savefig("velocity.jpeg")
plt.show()

