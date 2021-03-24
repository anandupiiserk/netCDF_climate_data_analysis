#!/usr/bin/env python
# coding: utf-8
cd ../../200_850_track
##################################################################################################
##################################################################################################
#################### calculation of wind shear from u and v components ###########################
##################################################################################################
##################################################################################################

import csv
import os                                             #for os.remove
from scipy.io import netcdf
from cdo import*
import random
import math
from numpy import nan                                 #this reads the nan values from the List of data
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


# In[4]:


#df = pd.read_excel('track.xls', sheetname='dat')
#d_cs = df.cs.values
#d_d = df.dep.values
########################################################################################
dtt = pd.read_excel('track_data.xlsx',sheetname = 'dat')
dt = dtt.date.values
LAT = dtt.lat.values
LON = dtt.lon.values
########################################################################################
N2 = [2,3,11,2,4,4,7,7,6,6,4,26,3,5,4,7,4,17,2,3,8,3,7,4,6,7,15,6,19,5,3,7,5,4,3,3,5,9]
#N1 = [3,4,12,3,5,5,8,8,7,7,5,27,4,6,5,8,5,18,3,4,9,4,8,5,7,8,16,7,20,6,4,8,6,5,4,4,6,10]
#M = ["%02d" % n1 for n1 in N2]
lat1 = LAT - 1.25
lat2 = LAT + 1.25
lon1 = LON - 1.25
lon2 = LON + 1.25
MM = [1,2,3,4,5,6,7,8,9,10,11,12]
M = ["%02d" % n1 for n1 in MM]


# The maximum value of sustained wind speed can be seen at the frst quadrant so we are taking the value
# in the first quadrant so that there will be a zone where Vs will be a value between 5m/s to 15m/s
# which is apt for the formation of a tropical cyclone.

#                                         200_v

# In[4]:


for i in range(len(dt)):
    dat1 = dt[i]
    Lat1 = lat1[i]
    Lat2 = lat2[i]
    Lon1 = lon1[i]
    Lon2 = lon2[i]
    Lev1 = 200
    dat1 = int(dat1)
    dat2 = dat1 + 1
    nan = nan
    filename = '200_v.nc'
    cdo.seldate(dat1, input = filename, output = str(dat1)+str(Lat1)+'.nc', options = '-f nc' )
    #seldate need two date ranges for the code to work and it includes both dates
    filename = str(dat1)+str(Lat1)+'.nc'
    cdo.sellevel(Lev1, input = filename , output = str(dat1)+str(Lon1)+'.nc', options = '-f nc')
    os.remove(str(dat1)+str(Lat1)+'.nc')
    filename = str(dat1)+str(Lon1)+'.nc'
    cdo.sellonlatbox(Lon1,Lon2,Lat1,Lat2, input = filename , output = str(dat1)+str(Lat1)+str(Lat2)+'.nc' , options = '-f nc')
    os.remove(str(dat1)+str(Lon1)+'.nc')
    X1 = []   #defining X1
    X2 = []   #defining X2
    filename = str(dat1)+str(Lat1)+str(Lat2)+'.nc'
    dataset = nc.Dataset(filename)
    dataset.variables.keys()
    dataset.dimensions.keys()
    open_netcdf = xr.open_dataset(filename)
    dataset = open_netcdf.to_dataframe()
    du = dataset.v.values
    os.remove(str(dat1)+str(Lat1)+str(Lat2)+'.nc')
    SS = []
    TT = []
    for i in range(len(du)):
        if du[i] == nan:
            SS.append(du[i])   #This is a dummy variable and SS remain as [] and nothing happens as nan wont be appended.
        else:
            TT.append(du[i])   #This is the most needed one and every values will be appended to this list.
    TT = [tt for tt in TT if str(tt) != 'nan']
    averd = np.average(TT) #averagind the value sin the grid
    X1.append(averd)    #appending the grid averaged values to X1 list
    x1 = np.asarray(X1) #appending X1 asarray in x1
    sstu = np.max(X1)   #only one value is there, so there is no problem with this value
    #K = []              #creating a list
    #K.append("The Cyclone located at " +str(Lat1)+"N "+str(Lon1)+"E and " +str(Lat2)+"N "+str(Lon2)+ "E on " +str(dat1)+" with u component:"+str(sstu))
    #np.savetxt("Cyclone on"+str(dat1)+".txt",K,fmt='%5s',delimiter=',')
    #np.savetxt("Daily grid avg for "+str(cycl)+".txt",SST,fmt='%5s',delimiter=',')
    f = open("v_200_track.txt","a+")
    f.write("%s\r\n" % sstu)   # %s is used for the decimal points and %d is used for the ones tenths sand other places
    f.close()       


# In[5]:


import csv
import os                                             #for os.remove
from scipy.io import netcdf
from cdo import*
import random
import math
from numpy import nan                                 #this reads the nan values from the List of data
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

#df = pd.read_excel('track.xls', sheetname='dat')
#d_cs = df.cs.values
#d_d = df.dep.values
########################################################################################
dtt = pd.read_excel('track_data.xlsx',sheetname = 'dat')
dt = dtt.date.values
LAT = dtt.lat.values
LON = dtt.lon.values
########################################################################################
N2 = [2,3,11,2,4,4,7,7,6,6,4,26,3,5,4,7,4,17,2,3,8,3,7,4,6,7,15,6,19,5,3,7,5,4,3,3,5,9]
#N1 = [3,4,12,3,5,5,8,8,7,7,5,27,4,6,5,8,5,18,3,4,9,4,8,5,7,8,16,7,20,6,4,8,6,5,4,4,6,10]
#M = ["%02d" % n1 for n1 in N2]
lat1 = LAT - 1.25
lat2 = LAT + 1.25
lon1 = LON - 1.25
lon2 = LON + 1.25
MM = [1,2,3,4,5,6,7,8,9,10,11,12]
M = ["%02d" % n1 for n1 in MM]


# In[6]:


for i in range(len(dt)):
    dat1 = dt[i]
    Lat1 = lat1[i]
    Lat2 = lat2[i]
    Lon1 = lon1[i]
    Lon2 = lon2[i]
    Lev1 = 200
    dat1 = int(dat1)
    dat2 = dat1 + 1
    nan = nan
    filename = '200_u.nc'
    cdo.seldate(dat1, input = filename, output = str(dat1)+str(Lat1)+'.nc', options = '-f nc' )
    #seldate need two date ranges for the code to work and it includes both dates
    filename = str(dat1)+str(Lat1)+'.nc'
    cdo.sellevel(Lev1, input = filename , output = str(dat1)+str(Lon1)+'.nc', options = '-f nc')
    os.remove(str(dat1)+str(Lat1)+'.nc')
    filename = str(dat1)+str(Lon1)+'.nc'
    cdo.sellonlatbox(Lon1,Lon2,Lat1,Lat2, input = filename , output = str(dat1)+str(Lat1)+str(Lat2)+'.nc' , options = '-f nc')
    os.remove(str(dat1)+str(Lon1)+'.nc')
    X1 = []   #defining X1
    X2 = []   #defining X2
    filename = str(dat1)+str(Lat1)+str(Lat2)+'.nc'
    dataset = nc.Dataset(filename)
    dataset.variables.keys()
    dataset.dimensions.keys()
    open_netcdf = xr.open_dataset(filename)
    dataset = open_netcdf.to_dataframe()
    du = dataset.u.values
    os.remove(str(dat1)+str(Lat1)+str(Lat2)+'.nc')
    SS = []
    TT = []
    for i in range(len(du)):
        if du[i] == nan:
            SS.append(du[i])   #This is a dummy variable and SS remain as [] and nothing happens as nan wont be appended.
        else:
            TT.append(du[i])   #This is the most needed one and every values will be appended to this list.
    TT = [tt for tt in TT if str(tt) != 'nan']
    averd = np.average(TT) #averagind the value sin the grid
    X1.append(averd)    #appending the grid averaged values to X1 list
    x1 = np.asarray(X1) #appending X1 asarray in x1
    sstu = np.max(X1)   #only one value is there, so there is no problem with this value
    #K = []              #creating a list
    #K.append("The Cyclone located at " +str(Lat1)+"N "+str(Lon1)+"E and " +str(Lat2)+"N "+str(Lon2)+ "E on " +str(dat1)+" with u component:"+str(sstu))
    #np.savetxt("Cyclone on"+str(dat1)+".txt",K,fmt='%5s',delimiter=',')
    #np.savetxt("Daily grid avg for "+str(cycl)+".txt",SST,fmt='%5s',delimiter=',')
    f = open("u_200_track.txt","a+")
    f.write("%s\r\n" % sstu)   # %s is used for the decimal points and %d is used for the ones tenths sand other places
    f.close()       


# In[7]:


import csv
import os                                             #for os.remove
from scipy.io import netcdf
from cdo import*
import random
import math
from numpy import nan                                 #this reads the nan values from the List of data
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

#df = pd.read_excel('track.xls', sheetname='dat')
#d_cs = df.cs.values
#d_d = df.dep.values
########################################################################################
dtt = pd.read_excel('track_data.xlsx',sheetname = 'dat')
dt = dtt.date.values
LAT = dtt.lat.values
LON = dtt.lon.values
########################################################################################
N2 = [2,3,11,2,4,4,7,7,6,6,4,26,3,5,4,7,4,17,2,3,8,3,7,4,6,7,15,6,19,5,3,7,5,4,3,3,5,9]
#N1 = [3,4,12,3,5,5,8,8,7,7,5,27,4,6,5,8,5,18,3,4,9,4,8,5,7,8,16,7,20,6,4,8,6,5,4,4,6,10]
#M = ["%02d" % n1 for n1 in N2]
lat1 = LAT - 1.25
lat2 = LAT + 1.25
lon1 = LON - 1.25
lon2 = LON + 1.25
MM = [1,2,3,4,5,6,7,8,9,10,11,12]
M = ["%02d" % n1 for n1 in MM]


# In[8]:


for i in range(len(dt)):
    dat1 = dt[i]
    Lat1 = lat1[i]
    Lat2 = lat2[i]
    Lon1 = lon1[i]
    Lon2 = lon2[i]
    Lev1 = 850
    dat1 = int(dat1)
    dat2 = dat1 + 1
    nan = nan
    filename = '850_u.nc'
    cdo.seldate(dat1, input = filename, output = str(dat1)+str(Lat1)+'.nc', options = '-f nc' )
    #seldate need two date ranges for the code to work and it includes both dates
    filename = str(dat1)+str(Lat1)+'.nc'
    cdo.sellevel(Lev1, input = filename , output = str(dat1)+str(Lon1)+'.nc', options = '-f nc')
    os.remove(str(dat1)+str(Lat1)+'.nc')
    filename = str(dat1)+str(Lon1)+'.nc'
    cdo.sellonlatbox(Lon1,Lon2,Lat1,Lat2, input = filename , output = str(dat1)+str(Lat1)+str(Lat2)+'.nc' , options = '-f nc')
    os.remove(str(dat1)+str(Lon1)+'.nc')
    X1 = []   #defining X1
    X2 = []   #defining X2
    filename = str(dat1)+str(Lat1)+str(Lat2)+'.nc'
    dataset = nc.Dataset(filename)
    dataset.variables.keys()
    dataset.dimensions.keys()
    open_netcdf = xr.open_dataset(filename)
    dataset = open_netcdf.to_dataframe()
    du = dataset.u.values
    os.remove(str(dat1)+str(Lat1)+str(Lat2)+'.nc')
    SS = []
    TT = []
    for i in range(len(du)):
        if du[i] == nan:
            SS.append(du[i])   #This is a dummy variable and SS remain as [] and nothing happens as nan wont be appended.
        else:
            TT.append(du[i])   #This is the most needed one and every values will be appended to this list.
    TT = [tt for tt in TT if str(tt) != 'nan']
    averd = np.average(TT) #averagind the value sin the grid
    X1.append(averd)    #appending the grid averaged values to X1 list
    x1 = np.asarray(X1) #appending X1 asarray in x1
    sstu = np.max(X1)   #only one value is there, so there is no problem with this value
    #K = []              #creating a list
    #K.append("The Cyclone located at " +str(Lat1)+"N "+str(Lon1)+"E and " +str(Lat2)+"N "+str(Lon2)+ "E on " +str(dat1)+" with u component:"+str(sstu))
    #np.savetxt("Cyclone on"+str(dat1)+".txt",K,fmt='%5s',delimiter=',')
    #np.savetxt("Daily grid avg for "+str(cycl)+".txt",SST,fmt='%5s',delimiter=',')
    f = open("u_850_track.txt","a+")
    f.write("%s\r\n" % sstu)   # %s is used for the decimal points and %d is used for the ones tenths sand other places
    f.close() 


# In[5]:


import csv
import os                                             #for os.remove
from scipy.io import netcdf
from cdo import*
import random
import math
from numpy import nan                                 #this reads the nan values from the List of data
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

#df = pd.read_excel('track.xls', sheetname='dat')
#d_cs = df.cs.values
#d_d = df.dep.values
########################################################################################
dtt = pd.read_excel('track_data.xlsx',sheetname = 'dat')
dt = dtt.date.values
LAT = dtt.lat.values
LON = dtt.lon.values
########################################################################################
N2 = [2,3,11,2,4,4,7,7,6,6,4,26,3,5,4,7,4,17,2,3,8,3,7,4,6,7,15,6,19,5,3,7,5,4,3,3,5,9]
#N1 = [3,4,12,3,5,5,8,8,7,7,5,27,4,6,5,8,5,18,3,4,9,4,8,5,7,8,16,7,20,6,4,8,6,5,4,4,6,10]
#M = ["%02d" % n1 for n1 in N2]
lat1 = LAT - 1.25
lat2 = LAT + 1.25
lon1 = LON - 1.25
lon2 = LON + 1.25
MM = [1,2,3,4,5,6,7,8,9,10,11,12]
M = ["%02d" % n1 for n1 in MM]


# In[6]:


for i in range(len(dt)):
    dat1 = dt[i]
    Lat1 = lat1[i]
    Lat2 = lat2[i]
    Lon1 = lon1[i]
    Lon2 = lon2[i]
    Lev1 = 850
    dat1 = int(dat1)
    dat2 = dat1 + 1
    nan = nan
    filename = '850_v.nc'
    cdo.seldate(dat1, input = filename, output = str(dat1)+str(Lat1)+'.nc', options = '-f nc' )
    #seldate need two date ranges for the code to work and it includes both dates
    filename = str(dat1)+str(Lat1)+'.nc'
    cdo.sellevel(Lev1, input = filename , output = str(dat1)+str(Lon1)+'.nc', options = '-f nc')
    os.remove(str(dat1)+str(Lat1)+'.nc')
    filename = str(dat1)+str(Lon1)+'.nc'
    cdo.sellonlatbox(Lon1,Lon2,Lat1,Lat2, input = filename , output = str(dat1)+str(Lat1)+str(Lat2)+'.nc' , options = '-f nc')
    os.remove(str(dat1)+str(Lon1)+'.nc')
    X1 = []   #defining X1
    X2 = []   #defining X2
    filename = str(dat1)+str(Lat1)+str(Lat2)+'.nc'
    dataset = nc.Dataset(filename)
    dataset.variables.keys()
    dataset.dimensions.keys()
    open_netcdf = xr.open_dataset(filename)
    dataset = open_netcdf.to_dataframe()
    du = dataset.v.values
    os.remove(str(dat1)+str(Lat1)+str(Lat2)+'.nc')
    SS = []
    TT = []
    for i in range(len(du)):
        if du[i] == nan:
            SS.append(du[i])   #This is a dummy variable and SS remain as [] and nothing happens as nan wont be appended.
        else:
            TT.append(du[i])   #This is the most needed one and every values will be appended to this list.
    TT = [tt for tt in TT if str(tt) != 'nan']
    averd = np.average(TT) #averagind the value sin the grid
    X1.append(averd)    #appending the grid averaged values to X1 list
    x1 = np.asarray(X1) #appending X1 asarray in x1
    sstu = np.max(X1)   #only one value is there, so there is no problem with this value
    #K = []              #creating a list
    #K.append("The Cyclone located at " +str(Lat1)+"N "+str(Lon1)+"E and " +str(Lat2)+"N "+str(Lon2)+ "E on " +str(dat1)+" with u component:"+str(sstu))
    #np.savetxt("Cyclone on"+str(dat1)+".txt",K,fmt='%5s',delimiter=',')
    #np.savetxt("Daily grid avg for "+str(cycl)+".txt",SST,fmt='%5s',delimiter=',')
    f = open("v_850_track.txt","a+")
    f.write("%s\r\n" % sstu)   # %s is used for the decimal points and %d is used for the ones tenths sand other places
    f.close()       


#                                         evaluating the values

# In[7]:


u200 = np.loadtxt("u_200_track.txt", dtype=float)      #reading the files produced from the previous codes
v200 = np.loadtxt("v_200_track.txt", dtype=float)
u850 = np.loadtxt("u_850_track.txt", dtype=float)
v850 = np.loadtxt("v_850_track.txt", dtype=float)


# In[8]:


value_1 = (u200**2 + v200**2)**0.5       #caclulating the u squared plus v squared for 200hpa
value_2 = (u850**2 + v850**2)**0.5       #calculating the u squared plus v squared for 850hpa
final_value = value_1-value_2            #caculating the difference in the value of 200-850


# In[9]:


np.savetxt("200-850_track.txt",final_value,fmt='%5s',delimiter=',')
#this file have the wind shear values




