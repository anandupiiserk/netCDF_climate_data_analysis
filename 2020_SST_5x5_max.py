#!/usr/bin/env python
# coding: utf-8

# In[1]:
##################################################################################################
##################################################################################################
############ calculation of sst value from the gridded global climate model of AVHRR #############
##################################################################################################
##################################################################################################
##################################### file format is netCDF ######################################
cd ../Datas/


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


df = pd.read_excel("sst_grid.xlsx",sheet_name='Sheet2')
date = df.date.values
lat = df.lat.values
lon = df.lon.values


# In[4]:


lat1 = lat + 2.5
lat2 = lat - 2.5
lon1 = lon + 2.5
lon2 = lon - 2.5


# cycl = input("Enter the name of Cyclone :")
# dat3 = input("Enter the ending date in YYYYMMDD format:")
# dat3 = int(dat3)
# dat2 = dat3+1
# dat1 = dat2-4
# Lat1 = input("Enter the starting latitude:")
# Lat2 = input("Enter the ending latitude:")
# Lon1 = input("Enter the starting longitude:")
# Lon2 = input("Enter the ending longitude:") 
# dat1 = int(dat1)         #converting strings to integers.
# dat2 = int(dat2)
# #Lat1 = int(Lat1)        #Here it is not needed becuase integers are not needed.
# #Lat2 = int(Lat2)
# #Lon1 = int(Lon1)
# #Lon2 = int(Lon2)
# 

# In[12]:


os.remove('max_sst_cs_max_sus.txt')
os.remove('5x5_sus_Max_temp.txt')
ssss = []
cs_sst_max = []
for i in range(len(date)):
    Lat1 = lat1[i]
    Lat2 = lat2[i]
    Lon1 = lon1[i]
    Lon2 = lon2[i]
    filename = 'avhrr-only-v2.'+str(date[i])+'.nc'#there are around 365*38 data files around which this code will loop through the requirede dates mentioned in the date file.
    cdo.sellonlatbox(Lon1,Lon2,Lat1,Lat2, input=filename , output = str(i)+'.nc', options = '-f nc')
    filename = str(i)+'.nc'  #reads the new cropped file
    dataset = nc.Dataset(filename)
    dataset.variables.keys()
    dataset.dimensions.keys()
    open_netcdf = xr.open_dataset(filename)
    dataset = open_netcdf.to_dataframe()
    dsst = dataset.sst.values #reading sst from the cropped section
    SS=[]
    TT=[]
    os.remove(str(i)+'.nc')
    SST=[]
    for i in range(len(dsst)):
        if dsst[i] == nan:
            SS.append(dsst[i])   #This is a dummy variable and SS remain as [] and nothing happens as nan wont be appended.
        else:
            TT.append(dsst[i])   #Everything gets appended to this and this will be a list with nan values.
    TT = [tt for tt in TT if str(tt) != 'nan'] #removing the nan values from the list without changing the index of the list.
    avg = np.average(TT)
    #SST.append(averd)		#appending the values to SST so that a list with SST daily averaged values of a 5x5 grid is obtained.
    #sstv = np.asarray(SST)
    #sstf = np.average(sstv)
    sstu = np.max(TT)
    ssss.append(avg)
    f = open("max_sst_cs_max_sus.txt","a+")
    f.write("%s\r\n" % [avg])   # %s is used for the decimal points and %d is used for the ones tenths sand other places
    f.close()
    ff = open("5x5_sus_Max_temp.txt","a+")
    ff.write("%s\r\n" % [sstu])
    # %s is used for the decimal points and %d is used for the ones tenths sand other places
    cs_sst_max.append(sstu) 
    ff.close()
    


# In[13]:


sst = []
for i in range(0,774,5):
    av = max(ssss[i],ssss[i+1],ssss[i+2],ssss[i+3],ssss[i+4])
    #av = avgg[i]+avgg[i+1]+avgg[i+2]+avgg[i+3]+avgg[i+4]
    #av = av/5
    sst.append(av)


# In[14]:


import xlsxwriter 

os.remove('sst_cs_5_max_Sus.xlsx') 

workbook = xlsxwriter.Workbook('sst_cs_5_max_Sus.xlsx') 
worksheet = workbook.add_worksheet() 
  
# Start from the first cell. 
# Rows and columns are zero indexed. 
row = 0
column = 0
  
content = sst
  
# iterating through content list 
for item in content : 
  
    # write operation perform 
    worksheet.write(row, column, item) 
  
    # incrementing the value of row by one 
    # with each iteratons. 
    row += 1
      
workbook.close() 


# In[15]:


sst = []
for i in range(0,774,5):
    av = max(cs_sst_max[i],cs_sst_max[i+1],cs_sst_max[i+2],cs_sst_max[i+3],cs_sst_max[i+4])
    #av = avgg[i]+avgg[i+1]+avgg[i+2]+avgg[i+3]+avgg[i+4]
    #av = av/5
    sst.append(av)


# In[16]:
#writing to an xlsx file for future use.

import xlsxwriter 

os.remove('sst_cs_5_max_Sus_max.xlsx') 

workbook = xlsxwriter.Workbook('sst_cs_5_max_Sus_max.xlsx') 
worksheet = workbook.add_worksheet() 
  
# Start from the first cell. 
# Rows and columns are zero indexed. 
row = 0
column = 0
  
content = sst
  
# iterating through content list 
for item in content : 
  
    # write operation perform 
    worksheet.write(row, column, item) 
  
    # incrementing the value of row by one 
    # with each iteratons. 
    row += 1
      
workbook.close() 


# In[ ]:




