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
import urllib
from datetime import datetime, timedelta
import urllib.request

LON1 = -68
LON2 = -51
LAT1 = -48
LAT2 = -39

# Uncomment to reprocess GFW effort and Seascape data
#--------------------------------------------------------------------------------

# OC_8D = ['2012-01-01', '2012-01-09', '2012-01-17', '2012-01-25', '2012-02-02',
#  '2012-02-10', '2012-02-18', '2012-02-26', '2012-03-05', '2012-03-13',
#  '2012-03-21', '2012-03-29', '2012-04-06', '2012-04-14', '2012-04-22',
#  '2012-04-30', '2012-05-08', '2012-05-16', '2012-05-24', '2012-06-01',
#  '2012-06-09', '2012-06-17', '2012-06-25', '2012-07-03', '2012-07-11',
#  '2012-07-19', '2012-07-27', '2012-08-04', '2012-08-12', '2012-08-20',
#  '2012-08-28', '2012-09-05', '2012-09-13', '2012-09-21', '2012-09-29',
#  '2012-10-07', '2012-10-15', '2012-10-23', '2012-10-31', '2012-11-08',
#  '2012-11-16', '2012-11-24', '2012-12-02', '2012-12-10', '2012-12-18',
#  '2012-12-26', '2013-01-01', '2013-01-09', '2013-01-17', '2013-01-25',
#  '2013-02-02', '2013-02-10', '2013-02-18', '2013-02-26', '2013-03-06',
#  '2013-03-14', '2013-03-22', '2013-03-30', '2013-04-07', '2013-04-15',
#  '2013-04-23', '2013-05-01', '2013-05-09', '2013-05-17', '2013-05-25',
#  '2013-06-02', '2013-06-10', '2013-06-18', '2013-06-26', '2013-07-04',
#  '2013-07-12', '2013-07-20', '2013-07-28', '2013-08-05', '2013-08-13',
#  '2013-08-21', '2013-08-29', '2013-09-06', '2013-09-14', '2013-09-22',
#  '2013-09-30', '2013-10-08', '2013-10-16', '2013-10-24', '2013-11-01',
#  '2013-11-09', '2013-11-17', '2013-11-25', '2013-12-03', '2013-12-11',
#  '2013-12-19', '2013-12-27', '2014-01-01', '2014-01-09', '2014-01-17',
#  '2014-01-25', '2014-02-02', '2014-02-10', '2014-02-18', '2014-02-26',
#  '2014-03-06', '2014-03-14', '2014-03-22', '2014-03-30', '2014-04-07',
#  '2014-04-15', '2014-04-23', '2014-05-01', '2014-05-09', '2014-05-17',
#  '2014-05-25', '2014-06-02', '2014-06-10', '2014-06-18', '2014-06-26',
#  '2014-07-04', '2014-07-12', '2014-07-20', '2014-07-28', '2014-08-05',
#  '2014-08-13', '2014-08-21', '2014-08-29', '2014-09-06', '2014-09-14',
#  '2014-09-22', '2014-09-30', '2014-10-08', '2014-10-16', '2014-10-24',
#  '2014-11-01', '2014-11-09', '2014-11-17', '2014-11-25', '2014-12-03',
#  '2014-12-11', '2014-12-19', '2014-12-27', '2015-01-01', '2015-01-09',
#  '2015-01-17', '2015-01-25', '2015-02-02', '2015-02-10', '2015-02-18',
#  '2015-02-26', '2015-03-06', '2015-03-14', '2015-03-22', '2015-03-30',
#  '2015-04-07', '2015-04-15', '2015-04-23', '2015-05-01', '2015-05-09',
#  '2015-05-17', '2015-05-25', '2015-06-02', '2015-06-10', '2015-06-18',
#  '2015-06-26', '2015-07-04', '2015-07-12', '2015-07-20', '2015-07-28',
#  '2015-08-05', '2015-08-13', '2015-08-21', '2015-08-29', '2015-09-06',
#  '2015-09-14', '2015-09-22', '2015-09-30', '2015-10-08', '2015-10-16',
#  '2015-10-24', '2015-11-01', '2015-11-09', '2015-11-17', '2015-11-25',
#  '2015-12-03', '2015-12-11', '2015-12-19', '2015-12-27', '2016-01-01',
#  '2016-01-09', '2016-01-17', '2016-01-25', '2016-02-02', '2016-02-10',
#  '2016-02-18', '2016-02-26', '2016-03-05', '2016-03-13', '2016-03-21',
#  '2016-03-29', '2016-04-06', '2016-04-14', '2016-04-22', '2016-04-30',
#  '2016-05-08', '2016-05-16', '2016-05-24', '2016-06-01', '2016-06-09',
#  '2016-06-17', '2016-06-25', '2016-07-03', '2016-07-11', '2016-07-19',
#  '2016-07-27', '2016-08-04', '2016-08-12', '2016-08-20', '2016-08-28',
#  '2016-09-05', '2016-09-13', '2016-09-21', '2016-09-29', '2016-10-07',
#  '2016-10-15', '2016-10-23', '2016-10-31', '2016-11-08', '2016-11-16',
#  '2016-11-24', '2016-12-02', '2016-12-10', '2016-12-18', '2016-12-26']

#Parse: GFW Effort Data -----------------------------------------------
# Get global fish watch data
# GFW_DIR = '/data2/GFW_public/fishing_effort_100d/daily_csvs'

# # # allFiles = sorted(glob.glob(GFW_DIR + "/*.csv"))

# list_ = []
# outdat = pd.DataFrame()
# # # Append files in subdir
# for i in range(len(OC_8D)):
#     file_ = f"{GFW_DIR}/{OC_8D[i]}.csv"
#     print(file_)
#     df = pd.read_csv(file_, index_col=None, header=0, low_memory=False)
#     df['lat1'] = df['lat_bin']/100
#     df['lon1'] = df['lon_bin']/100
#     df['lat2'] = df['lat1'] + 0.010
#     df['lon2'] = df['lon1'] + 0.010
#     dat = df[(df['lon1'] >= LON1) & (df['lon2'] <= LON2) & (df['lat1'] >= LAT1) & (df['lat2'] <= LAT2)] 
#     list_.append(dat)
#     outdat = pd.concat(list_, axis = 0, ignore_index = True)
#     print(OC_8D[i])

# outdat.to_feather("data/patagonia_shelf_gfw_effort_100d_data.feather")



# Parse: Seascape Data -----------------------------------------------

# https://cwcgom.aoml.noaa.gov/thredds/ncss/SEASCAPE_8DAY/SEASCAPES.nc?var=CLASS&var=P&north=-39&west=-68&east=-51&south=-48&disableProjSubset=on&horizStride=1&time_start=2012-01-01T12%3A00%3A00Z&time_end=2016-12-31T12%3A00%3A00Z&timeStride=1&addLatLon=true&accept=netcdf

# url = f"https://cwcgom.aoml.noaa.gov/thredds/ncss/SEASCAPE_8DAY/SEASCAPES.nc?var=CLASS&var=P&north=-39&west=-68&east=-51&south=-48&disableProjSubset=on&horizStride=1&time_start=2012-01-01T12%3A00%3A00Z&time_end=2016-12-31T12%3A00%3A00Z&timeStride=1&addLatLon=true&accept=netcdf"    

# # Download classes
# urllib.request.urlretrieve(url, filename = f"data/seascapes/seascapes_8D_CLASS_PROB_2012-2016.nc")    


# file = "data/seascapes/seascapes_8D_CLASS_PROB_2012-2016.nc"

# ds = xr.open_dataset(file)
# df = ds.to_dataframe().reset_index()
# df = df[(df['lon'] >= LON1) & (df['lon'] <= LON2)]
# df = df[(df['lat'] >= LAT1) & (df['lat'] <= LAT2)]
# df = df.reset_index(drop=True)

# # Issues with zero being letter O in seascape data
# df['date'] = df['time'].apply(lambda x: f"{x.year}" + f"-{x.month}".zfill(3) + f"-{x.day}".zfill(3))
# df = df[['date', 'lon', 'lat', 'CLASS', 'P']]
# df.columns = ['date', 'lon', 'lat', 'seascape_class', 'seascape_prob']
# df.to_feather('data/patagonia_shelf_seascapes_2012-2016.feather')

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

# files = glob.glob('/data2/CHL/NC/8DAY/*.nc')

# rdat = pd.DataFrame()
# for file_ in files:
#     ds = xr.open_dataset(file_, drop_variables=['palette'])
#     df = ds.to_dataframe().reset_index()
#     df = df[(df['lon'] >= LON1) & (df['lon'] <= LON2)]
#     df = df[(df['lat'] >= LAT1) & (df['lat'] <= LAT2)]
#     df['date'] = ds.time_coverage_start
#     year = pd.DatetimeIndex(df['date'])[0].year
#     month = pd.DatetimeIndex(df['date'])[0].month
#     day = pd.DatetimeIndex(df['date'])[0].day
#     df['date'] = f"{year}" + f"-{month}".zfill(3) + f"-{day}".zfill(3)
#     df = df[['date', 'lon', 'lat', 'chlor_a']]
#     rdat = pd.concat([rdat, df])
#     print(f"{year}" + f"-{month}".zfill(3) + f"-{day}".zfill(3))

# rdat = rdat.reset_index()
# rdat = rdat[['date', 'lon', 'lat', 'chlor_a']]
# rdat.to_feather('data/patagonia_shelf_CHL_2012-2016.feather')

#--------------------------------------------------------------------------------

# Load processed files

# gfw data
# lat_bin southern edge of grid
# lon_bin western edge of grid
# lat2 northern edge
# lon2 eastern edge
gfw = pd.read_feather("data/patagonia_shelf_gfw_effort_100d_data.feather")

# seascape data
sea = pd.read_feather("data/patagonia_shelf_seascapes_2012-2016.feather")

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

#gfw = gfw[(gfw['year'] == 2016) & (gfw['month'] == 1) & (gfw['day'] == 1)]
#sea = sea[(sea['year'] == 2016) & (sea['month'] == 1) & (sea['day'] == 1)]


#len(gfw)
#len(sea)

#gfw = gfw.head(50)
#sea = sea.head(50)


def dist(lat1, lon1, lat2, lon2):
    return np.sqrt( (lat2 - lat1)**2 + (lon2 - lon1)**2)

def find_seascape(lat, lon):

    #lat = -47
    #lon = -67

    lat1 = lat - .5
    lat2 = lat + .5
    lon1 = lon - .5
    lon2 = lon + .5

    indat = sea[(sea['lon'].values >= lon1) & (sea['lon'].values <= lon2) & (sea['lat'].values >= lat1) & (sea['lat'].values <= lat2)] 
    
    distances = indat.apply(
        lambda row: dist(lat, lon, row['lat'], row['lon']), 
        axis=1)
    
    rdat = pd.DataFrame({
        "seascape_class": [indat.loc[distances.idxmin(), 'seascape_class']],
        "seascape_prob": [indat.loc[distances.idxmin(), 'seascape_prob']]
    })
    #print(rdat)
    #return rdat
    return (indat.loc[distances.idxmin(), 'seascape_class'], indat.loc[distances.idxmin(), 'seascape_prob'])

def find_sst(lat, lon):

    lat1 = lat - .5
    lat2 = lat + .5
    lon1 = lon - .5
    lon2 = lon + .5

    indat = sst[(sst['lon'].values >= lon1) & (sst['lon'].values <= lon2) & (sst['lat'].values >= lat1) & (sst['lat'].values <= lat2)] 
    
    distances = indat.apply(
        lambda row: dist(lat, lon, row['lat'], row['lon']), 
        axis=1)
    print(indat.loc[distances.idxmin(), ['sst', 'lon', 'lat']])
    return indat.loc[distances.idxmin(), 'sst']

def find_chlor(lat, lon):

    lat1 = lat - .5
    lat2 = lat + .5
    lon1 = lon - .5
    lon2 = lon + .5

    indat = chl[(chl['lon'].values >= lon1) & (chl['lon'].values <= lon2) & (chl['lat'].values >= lat1) & (chl['lat'].values <= lat2)] 
    
    distances = indat.apply(
        lambda row: dist(lat, lon, row['lat'], row['lon']), 
        axis=1)
    
    
    return indat.loc[distances.idxmin(), 'chlor_a']

#@ray.remote
def process_days(dat):
    date = dat['date'].iat[0]
    #print(f"Processing data for: {date}")

    #print("1-Linking Effort and Seascape")
    # Link seascape to effort
    #dat.loc[:, 'seascape_class'], dat.loc[:, 'seascape_prob'] = zip(*dat.apply(lambda row: find_seascape(row['lat2'], row['lon2']), axis=1))
    # zip(*df_test['size'].apply(sizes))
    
    #print("2-Linking Effort and SST")
    # Link sst to effort
    dat.loc[:, 'sst'] = dat.apply(lambda row: find_sst(row['lat2'], row['lon2']), axis=1)
    
    #print("3-Linking Effort and CHL")
    # Link sst to effort
    #dat.loc[:, 'chlor_a'] = dat.apply(lambda row: find_chlor(row['lat2'], row['lon2']), axis=1)

    #print(f"4-Save data to data/processed/processed_{date}.feather")
    # Save data
    outdat = dat.reset_index(drop=True)
    #outdat.to_feather(f"data/processed/processed_{date}.feather")
    #print(f"{date}: COMPLETE")

    return outdat


gb = gfw.groupby('date')
days = [gb.get_group(x) for x in gb.groups]

# Debug
days = days[0]

#days[0] = days[0].loc[0:3, :]
#days[1] = days[1].loc[0:3, :]

days = days.loc[1:3, :]

#dat = days.loc[1:3, :]
test = process_days(days)
test2 = sst[sst.date == '2012-01-01']

test2.head()

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
files = glob.glob('data/processed/*.feather')
files
list_ = []
for file in files:
    df = pd.read_feather(file)
    list_.append(df)
    mdat = pd.concat(list_, sort=False)


mdat = mdat.reset_index(drop=True)
mdat.to_feather('data/full_gfw_effort_model_data_8DAY_2012-01-01_2016-12-26.feather')






