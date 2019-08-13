#import ray
import pandas as pd
import os
import glob
import netCDF4
from netCDF4 import Dataset
from scipy import spatial
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from datetime import datetime
import xarray as xr
from tqdm import tqdm, tqdm_pandas, tqdm_notebook
tqdm.pandas()

LON1 = -68
LON2 = -51
LAT1 = -48
LAT2 = -39

# Uncomment to reprocess GFW effort and Seascape data
#--------------------------------------------------------------------------------

# Parse: Seascape Data -----------------------------------------------

# SEASCAPE_DIR = 'data/seascapes/class'

# allFiles = sorted(glob.glob(SEASCAPE_DIR + "/*.csv"))

# list_ = []

# for file_ in allFiles:
#     print(file_)
#     df = pd.read_csv(file_, index_col=None, header=1, low_memory=False)
#     list_.append(df)
#     outdat = pd.concat(list_, axis = 0, ignore_index = True)

# outdat['date'] = outdat['UTC'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT12:00:00Z'))

# outdat = outdat[['date', 'degrees_north', 'degrees_east', 'None']]
# outdat = outdat[pd.DatetimeIndex(outdat['date']).year < 2017]

# unique_dates = outdat['date'].apply(lambda x: x.strftime("%Y-%m-%d"))
# unique_dates = unique_dates.unique()

# outdat = outdat.reset_index()
# outdat.to_feather("data/noaa_seascape_Patagonia_Shelf_2012-2016.feather")

#outdat = pd.read_feather("data/noaa_seascape_Patagonia_Shelf_2012-2016.feather")


# Parse: GFW Effort Data -----------------------------------------------
# # Get global fish watch data
# GFW_DIR = '/data2/GFW_public/fishing_effort_100d/daily_csvs'

# # # allFiles = sorted(glob.glob(GFW_DIR + "/*.csv"))

# list_ = []
# outdat = pd.DataFrame()
# # # Append files in subdir
# for i in range(len(unique_dates)):
#     file_ = f"{GFW_DIR}/{unique_dates[i]}.csv"
#     print(file_)
#     df = pd.read_csv(file_, index_col=None, header=0, low_memory=False)
#     df['lat1'] = df['lat_bin']/100
#     df['lon1'] = df['lon_bin']/100
#     df['lat2'] = df['lat1'] + 0.010
#     df['lon2'] = df['lon1'] + 0.010
#     dat = df[(df['lon1'] >= LON1) & (df['lon2'] <= LON2) & (df['lat1'] >= LAT1) & (df['lat2'] <= LAT2)] 
#     list_.append(dat)
#     outdat = pd.concat(list_, axis = 0, ignore_index = True)
#     print(unique_dates[i])

# outdat.to_feather("data/patagonia_shelf_gfw_effort_100d_data.feather")

# Parse: SST Temperature -----------------------------------------------

# files = glob.glob('/data2/SST/8DAY/*.nc')

# rdat = pd.DataFrame()
# for file_ in files:
#     ds = xr.open_dataset(file_, drop_variables=['qual_sst', 'palette'])
#     df = ds.to_dataframe().reset_index()
#     df = df[(df['lon'] >= LON1) & (df['lon'] <= LON2)]
#     df = df[(df['lat'] >= LAT1) & (df['lat'] <= LAT2)]
#     df['date'] = ds.time_coverage_start
#     year = pd.DatetimeIndex(df['date'])[0].year
#     month = pd.DatetimeIndex(df['date'])[0].month
#     day = pd.DatetimeIndex(df['date'])[0].day
#     df['date'] = f"{year}" + f"-{month}".zfill(3) + f"-{day}".zfill(3)
#     df = df[['date', 'lon', 'lat', 'sst']]
#     rdat = pd.concat([rdat, df])
#     print(f"{year}" + f"-{month}".zfill(3) + f"-{day}".zfill(3))

# rdat = rdat.reset_index()
# rdat = rdat[['date', 'lon', 'lat', 'sst']]
# rdat.to_feather('data/patagonia_shelf_SST_2012-2016.feather')


# Parse: CHL -----------------------------------------------

files = glob.glob('/data2/CHL/NC/8DAY/*.nc')

rdat = pd.DataFrame()
for file_ in files:
    ds = xr.open_dataset(file_, drop_variables=['palette'])
    df = ds.to_dataframe().reset_index()
    df = df[(df['lon'] >= LON1) & (df['lon'] <= LON2)]
    df = df[(df['lat'] >= LAT1) & (df['lat'] <= LAT2)]
    df['date'] = ds.time_coverage_start
    year = pd.DatetimeIndex(df['date'])[0].year
    month = pd.DatetimeIndex(df['date'])[0].month
    day = pd.DatetimeIndex(df['date'])[0].day
    df['date'] = f"{year}" + f"-{month}".zfill(3) + f"-{day}".zfill(3)
    df = df[['date', 'lon', 'lat', 'chlor_a']]
    rdat = pd.concat([rdat, df])
    print(f"{year}" + f"-{month}".zfill(3) + f"-{day}".zfill(3))

rdat = rdat.reset_index()
rdat = rdat[['date', 'lon', 'lat', 'chlor_a']]
rdat.to_feather('data/patagonia_shelf_CHL_2012-2016.feather')

#--------------------------------------------------------------------------------

# Load processed files

# gfw data
# lat_bin southern edge of grid
# lon_bin western edge of grid
# lat2 northern edge
# lon2 eastern edge
gfw = pd.read_feather("data/patagonia_shelf_gfw_effort_100d_data.feather")

# seascape data
sea = pd.read_feather("data/noaa_seascape_Patagonia_Shelf_2012-2016.feather")

# sea surface temp
sst = pd.read_feather('data/patagonia_shelf_SST_2012-2016.feather')

# Chlor
chl = pd.read_feather('data/patagonia_shelf_CHL_2012-2016.feather')

#gfw.head()
#sea.head()
#sst.head()



# For each day
# in each year
# Find seascape for each fishing_hours

gfw['year'] = pd.DatetimeIndex(gfw['date']).year
sea['year'] = pd.DatetimeIndex(sea['date']).year
sst['year'] = pd.DatetimeIndex(sst['date']).year
chl['year'] = pd.DatetimeIndex(chl['date']).year

gfw['month'] = pd.DatetimeIndex(gfw['date']).month
sea['month'] = pd.DatetimeIndex(sea['date']).month
sst['month'] = pd.DatetimeIndex(sst['date']).month
chl['month'] = pd.DatetimeIndex(chl['date']).month

gfw['day'] = pd.DatetimeIndex(gfw['date']).day
sea['day'] = pd.DatetimeIndex(sea['date']).day
sst['day'] = pd.DatetimeIndex(sst['date']).day
chl['day'] = pd.DatetimeIndex(chl['date']).day

#dat1 = gfw[(gfw['year'] == 2016) & (gfw['month'] == 1) & (gfw['day'] == 1)]
#dat2 = sea[(sea['year'] == 2016) & (sea['month'] == 1) & (sea['day'] == 1)]

# NO IDEA WHY?!?!?!?!?!?!?
dat1 = gfw[(gfw['year'] != 2016)]
dat2 = sea[(sea['year'] != 2016)]
dat3 = sst[(sst['year'] != 2016)]
dat4 = chl[(chl['year'] != 2016)]

#len(dat1)
#len(dat2)

#dat1 = dat1.head(50)
#dat2 = dat2.head(50)

#dat = dat1[dat1['date'] == "2016-01-01"]

def dist(lat1, lon1, lat2, lon2):
    return np.sqrt( (lat2 - lat1)**2 + (lon2 - lon1)**2)

def find_seascape(lat, lon):

    #lat = -47
    #lon = -67

    lat1 = lat - .5
    lat2 = lat + .5
    lon1 = lon - .5
    lon2 = lon + .5

    indat = dat2[(dat2['degrees_east'].values >= lon1) & (dat2['degrees_east'].values <= lon2) & (dat2['degrees_north'].values >= lat1) & (dat2['degrees_north'].values <= lat2)] 
    
    distances = indat.apply(
        lambda row: dist(lat, lon, row['degrees_north'], row['degrees_east']), 
        axis=1)
    
    return indat.loc[distances.idxmin(), 'None']

def find_sst(lat, lon):

    lat1 = lat - .5
    lat2 = lat + .5
    lon1 = lon - .5
    lon2 = lon + .5

    indat = dat3[(dat3['lon'].values >= lon1) & (dat3['lon'].values <= lon2) & (dat3['lat'].values >= lat1) & (dat3['lat'].values <= lat2)] 
    
    distances = indat.apply(
        lambda row: dist(lat, lon, row['lat'], row['lon']), 
        axis=1)
    
    return indat.loc[distances.idxmin(), 'sst']

def find_chlor(lat, lon):

    lat1 = lat - .5
    lat2 = lat + .5
    lon1 = lon - .5
    lon2 = lon + .5

    indat = dat4[(dat4['lon'].values >= lon1) & (dat4['lon'].values <= lon2) & (dat4['lat'].values >= lat1) & (dat4['lat'].values <= lat2)] 
    
    distances = indat.apply(
        lambda row: dist(lat, lon, row['lat'], row['lon']), 
        axis=1)
    
    return indat.loc[distances.idxmin(), 'chlor_a']

#@ray.remote
def process_days(dat):
    date = dat['date'].iat[0]
    print(f"Processing data for: {date}")

    print("1-Linking Effort and Seascape")
    # Link seascape to effort
    dat['seascape'] = dat.apply(lambda row: find_seascape(row['lat2'], row['lon2']), axis=1)
    
    print("2-Linking Effort and SST")
    # Link sst to effort
    dat['sst'] = dat.apply(lambda row: find_sst(row['lat2'], row['lon2']), axis=1)
    
    print("3-Linking Effort and CHL")
    # Link sst to effort
    dat['chlor_a'] = dat.apply(lambda row: find_chlor(row['lat2'], row['lon2']), axis=1)

    print(f"4-Save data to data/processed/processed_{date}.feather")
    # Save data
    outdat = dat.reset_index(drop=True)
    outdat.to_feather(f"data/processed/processed_{date}.feather")
    print(f"{date}: COMPLETE")


gb = dat1.groupby('date')
days = [gb.get_group(x) for x in gb.groups]

days = days[0]

dat = days

test = process_days(days)

#ray.init()

#1626474458
#1000000000

#results = ray.get([process_days.remote(i) for i in days])
#
#ray.shutdown()

#pool = multiprocessing.Pool(50, maxtasksperchild=1)         
pool = multiprocessing.Pool(50)
pool.map(process_days, days)
pool.close()

# Combine processed files
# files = glob.glob('data/processed/*.feather')
# files
# list_ = []
# for file in files:
#     df = pd.read_feather(file)
#     list_.append(df)
#     mdat = pd.concat(list_, sort=False)


# mdat = mdat.reset_index()
# mdat.to_feather('data/full_gfw_seascape_CLASS_2012-01-01_2016-12-18.feather')






