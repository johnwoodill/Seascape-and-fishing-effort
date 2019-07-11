import ray
import pandas as pd
import os
import glob
import netCDF4
from netCDF4 import Dataset
from scipy import spatial
import numpy as np
from joblib import Parallel, delayed
import multiprocessing




lon1 = -68
lon2 = -51
lat1 = -48
lat2 = -39

# Uncomment to reprocess data
#--------------------------------------------------------------------------------


# # Get global fish watch data
# GFW_DIR = '/data2/GFW_public/fishing_effort_100d/daily_csvs'

# allFiles = sorted(glob.glob(GFW_DIR + "/*.csv"))

# list_ = []
# # Append files in subdir
# for file_ in allFiles:
#     print(file_)
#     df = pd.read_csv(file_, index_col=None, header=0, low_memory=False)
#     df['lat1'] = df['lat_bin']/10
#     df['lon1'] = df['lon_bin']/10
#     df['lat2'] = df['lat1'] + 0.010
#     df['lon2'] = df['lon1'] + 0.010
#     dat = df[(df['lon1'] >= lon1) & (df['lon2'] <= lon2)] 
#     dat = dat[(dat['lat1'] >= lat1) & (dat['lat2'] <= lat2)]
#     list_.append(dat)
#     outdat = pd.concat(list_, axis = 0, ignore_index = True)

# outdat.to_feather("data/patagonia_shelf_gfw_effort_100d_data.feather")

# # Get seascape data
# SEASCAPE_DIR = 'data/seascapes'

# allFiles = sorted(glob.glob(SEASCAPE_DIR + "/*.csv"))

# list_ = []

# for file_ in allFiles:
#     print(file_)
#     df = pd.read_csv(file_, index_col=None, header=1, low_memory=False)
#     list_.append(df)
#     outdat = pd.concat(list_, axis = 0, ignore_index = True)

#outdat['date'] = outdat['UTC'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT12:00:00Z'))

#outdat.to_feather("data/noaa_seascape_Patagonia_Shelf_2012-2016.feather")



#--------------------------------------------------------------------------------

# Load processed files

# gfw data
# lat_bin southern edge of grid
# lon_bin western edge of grid
# lat2 northern edge
# lon2 eastern edge
gfw = pd.read_feather("data/patagonia_shelf_gfw_effort_data.feather")

# seascape data
sea = pd.read_feather("data/noaa_seascape_Patagonia_Shelf_2012-2016.feather")

#gfw.head()
#sea.head()


# For each day
# in each year
# Find seascape for each fishing_hours

gfw['year'] = pd.DatetimeIndex(gfw['date']).year
sea['year'] = pd.DatetimeIndex(sea['date']).year

gfw['month'] = pd.DatetimeIndex(gfw['date']).month
sea['month'] = pd.DatetimeIndex(sea['date']).month

gfw['day'] = pd.DatetimeIndex(gfw['date']).day
sea['day'] = pd.DatetimeIndex(sea['date']).day


#dat1 = gfw[(gfw['year'] == 2016) & (gfw['month'] == 1) & (gfw['day'] == 1)]
#dat2 = sea[(sea['year'] == 2016) & (sea['month'] == 1) & (sea['day'] == 1)]

dat1 = gfw[(gfw['year'] == 2016)]
dat2 = sea[(sea['year'] == 2016)]


#len(dat1)
#len(dat2)

#dat1 = dat1.head(50)
#dat2 = dat2.head(50)

#dat = dat1[dat1['date'] == "2016-01-01"]

def dist(lat1, lon1, lat2, lon2):
    return np.sqrt( (lat2 - lat1)**2 + (lon2 - lon1)**2)

def find_seascape(lat, lon):

    lat1 = lat - 1
    lat2 = lat + 1
    lon1 = lon - 1
    lon2 = lon + 1

    indat = dat2[(dat2['degrees_east'] >= lon1) & (dat2['degrees_east'] <= lon2)] 
    indat = indat[(indat['degrees_north'] >= lat1) & (indat['degrees_north'] <= lat2)]

    distances = indat.apply(
        lambda row: dist(lat, lon, row['degrees_north'], row['degrees_east']), 
        axis=1)
    
    return indat.loc[distances.idxmin(), 'None']
    
def process_days(dat):
    date = dat['date'].iat[0]
    dat['seascape'] = dat.apply(lambda row: find_seascape(row['lat2'], row['lon2']), axis=1)
    outdat = dat.reset_index(drop=True)
    outdat.to_feather(f"data/process_CLASS/process_gfw_seascape_CLASS_2016_{date}.feather")
    print(date)

gb = dat1.groupby('date')
days = [gb.get_group(x) for x in gb.groups]

pool = multiprocessing.Pool(40, maxtasksperchild=1)         
pool.map(process_days, days)
pool.close()


#print("Processing 2016 Seascapes")

#dat1['seascape'] = dat1.apply(lambda row: find_seascape(row['date'], row['lat2'], row['lon2']), axis=1)

#outdat = dat1.reset_index(drop=True)

#dat1.groupby('seascape')['fishing_hours'].sum()

#print("Saving file to data/process_gfw_seascape_CLASS_2016_test.feather")

#outdat.to_feather('data/process_gfw_seascape_CLASS_2016_test.feather')

#test = pd.read_feather('data/process_gfw_seascape_2016.feather')
