#!/usr/bin/env python
# coding: utf-8

# In[1]:

##################################################################################################
##################################################################################################
####### calculation of Accumulated Cyclone energy from simple xlsx file using equations ##########
##################################################################################################
##################################################################################################
##################################################################################################
cd ../ACE


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


dt = pd.read_excel('data_ACE_calculation.xlsx',sheetname = 'Sheet1')
df = pd.read_excel('data_ACE_calculation.xlsx',sheetname = 'Sheet2')
dg = pd.read_excel('data_ACE_calculation.xlsx',sheetname = 'Sheet3')
freq = df.freq.values #frequency of occurance of cyclones
cat = dg.category.values
time = dt.time.values #time of of duratio occurance of cyclone
V = dt.V.values       #Maximum sustained wind speed in knots
terms = dg.terms.values #number of intervals of each duration
year = dg.year.values   #year of occurance of cyclones


# # creating a 43 elemented array(43 cyclones)

# fre = [0, 10,  13,  17,  26,  41,  51,  70,  83,  87,  97, 117, 120, 138,164, 215, 230, 236, 245, 259, 297, 311, 326, 344, 383, 385, 391,399, 434, 439, 445, 464, 500, 529, 575, 613, 633, 667, 718, 769,786, 839, 899, 909]

# In[4]:


SUM = []                         #we are slicing the values to each cyclonic duration into 43 arrays
for i in range(0,len(freq)-1):
    S = time[freq[i]:freq[i+1]]
    SUM.append(S)


# In[5]:


TTT = []   #dummy variable
for i in range(len(terms)):
    KH = SUM[i]                  #dummy variable
    for j in range(len(KH)-1):
        diff = KH[j+1]-KH[j]     #calculating the difference in the time in hours
        TTT.append(diff)         


# In[6]:


for n,i in enumerate(TTT): #there are non uniform intervals hence 0000-1200 = "-12" similarly corrected others
    if i == -1200:
        TTT[n] = 1200
    if i == -900:
        TTT[n] = 1500
    if i == -1800:
        TTT[n] = 600
    if i == -2100:
        TTT[n] = 300


# In[7]:


TERMS = terms-1
kkk = 0     #dummy variable
fre = []    #dummy variable
for i in range(len(TERMS)):
    kkk += TERMS[i]                #running average to make the intervals uniform
    fre.append(kkk)
fre.insert(0,0)       #inserting zero in the first index


# now fre and TTT are the required parameters and we need to calculate the duration of each cyclone

# In[8]:


ght = []
ggg = 0
for i in range(len(fre)-1):
    ggg = TTT[fre[i]:fre[i+1]]  #slicing the TTT duration to arrays with -1 values from each
    ght.append(ggg)


# In[9]:


ggg = 0
Y = []
for i in range(len(ght)):
    ggg = np.sum(ght[i])
    Y.append(ggg)


# In[10]:


y = []
for i in range(len(Y)):
    z = Y[i]/100
    y.append(z)


# In[11]:


S=[]
for i in range(len(y)):
        S.append(i)


# In[12]:


z = np.polyfit(S,y,1)


# In[13]:


Z = []
for i in range(len(S)):
    dg = z[0]*S[i]+z[1]
    Z.append(dg)


# In[14]:


labels = ["C%d" % i for i in range(len(S))]
# Setting the background color
Fig,ax = plt.figure(figsize=(20,7),facecolor='gainsboro')
ax.grid(zorder=2)
ax = plt.axes()
plt.rc('grid', linestyle="-", color='black')
ax.set_facecolor("seashell")
plt.rc('grid', linestyle="-", color='black')

plt.xlabel("Cyclones",fontsize='20',weight='bold')
plt.ylabel("Duration (in hrs)",fontsize='20',weight='bold')

plt.bar(S,y,color='dodgerblue',label='duration')
plt.plot(S,Z,color='black',linestyle='dashed',label='trend')

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
ax.legend()
for i in range(len(S)):
    plt.annotate(str(year[i]), 
                 xy=(S[i],0),
                 va="top",
                 ha="center",fontsize='14',rotation='90',weight='bold')
    plt.annotate(str(cat[i]), 
                 xy=(S[i]+0.05,y[i]), #definfing the place where you need the annotations
                 va="bottom",color='black',
                 ha="center",fontsize='14',rotation='90',weight='bold')    
    
plt.savefig("duration_of_cyclones.jpeg")


# array([ 1.54258532, 35.62896406])

# In[15]:


HH = []                     #Slicing to 43 arrays
for i in range(len(terms)):
    sh = V[freq[i]:freq[i+1]]
    HH.append(sh)


# In[16]:


avrg = []
for i in range(len(terms)):
    ttth = []  #dummy variable
    for j in range(terms[i]-1):
        apt = ((HH[i][j]+HH[i][j+1])/2)**2   #avrg gives the averaged value squared
        ttth.append(apt)
    avrg.append(ttth)


# In[17]:


Vm = []    
for i in range(len(terms)):
    ttth = []  #dummy variable
    for j in range(terms[i]-2):
        apt = ght[i][j]*avrg[i][j]   #avrg gives the averaged value squared
        ttth.append(apt)
    Vm.append(ttth)


# In[18]:


ttth


# In[19]:


ddd = 0
ace = []
for i in range(len(Vm)):
    ddd = np.sum(Vm[i])
    ace.append(ddd)


# In[20]:


AcE = []
for i in range(len(ace)):
    z = ace[i]/100
    AcE.append(z)


# # Equation for Accumulated cyclone Energy         $ACE = 10^{-4} \Sigma {V_{max}^{2}}$

# In[21]:


ACE = []
for i in range(len(AcE)):
    z = AcE[i]/10000
    ACE.append(z)


# In[22]:


z = np.polyfit(S,ACE,1)
Z = []
for i in range(len(S)):
    dg = z[0]*S[i]+z[1]
    Z.append(dg)


# In[23]:


z


# In[24]:


labels = ["C%d" % i for i in range(len(S))]
# Setting the background color
Fig = plt.figure(figsize=(20,8),facecolor='gainsboro')
plt.rcParams['axes.xmargin'] = 0.005
plt.rcParams['axes.ymargin'] = 0
ax = plt.axes()
ax.set_facecolor("seashell")
ax.grid(zorder=2)

plt.xlabel("Cyclones",labelpad=50,fontsize='20',weight='bold')
plt.ylabel("ACE ($10^4 Kn^2$)",fontsize='20',weight='bold')

plt.bar(S,ACE,color='dimgrey',label='Accumulated cyclone Energy')
plt.plot(S,Z,color='black',linestyle='dashed',label='trend = 1.05x10$^{4}$kn$^{2}$')

#plt.bar(S,Min_val,color='lightcoral',label='Convergence',width=0.67)
#plt.plot(S,Zmn,color='midnightblue',linestyle='dotted')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)

#plt.yticks(fontsize='15',weight='bold')
plt.yticks(np.arange(0, 180, 20),fontsize='15',weight='bold')
plt.xticks(fontsize='15',weight='bold')
#plt.ylim(-20,180)
ax.legend(loc='upper left',fontsize='15')
for i in range(len(S)): #length of x axis
    plt.annotate(str(year[i]), 
                 xy=(S[i],0),
                 va="top",
                 ha="center",fontsize='14',rotation='90',weight='bold')

plt.title("Accumulated Cyclone Energy per cyclone over the Arabian Sea(1982-2020)",weight='bold', fontsize='20')    
plt.savefig("ACE.jpeg")



