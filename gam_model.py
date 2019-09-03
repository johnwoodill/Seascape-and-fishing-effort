import pandas as pd
import numpy as np 
from pygam import LinearGAM, s, f


dat = pd.read_feather('data/full_gfw_10d_effort_model_data_8DAY_2012-01-01_2016-12-26.feather')

#dat = dat[dat.geartype == 'drifting_longlines']
#dat = dat.sort_values('date')

# ~50% of obs are zero (remove?)
len(dat[dat.fishing_hours > 0])/len(dat)
dat = dat[dat.fishing_hours > 0]


# Linear model
# Get data frame of variables and dummy seascapes
moddat = dat[['fishing_hours', 'flag', 'year', 'month', 'seascape_class', 'sst', 'sst_grad', 'sst4', 'sst4_grad', 'chlor_a', 'lon1', 'lat1', 'tonnage', 'depth_m', 'coast_dist_km', 'port_dist_km']].dropna().reset_index(drop=True)

# Square sst
#moddat['sst_sq'] = moddat['sst']**2
#moddat = moddat[['fishing_hours', 'month', 'seascape_class', 'sst', 'sst_sq', 'chlor_a', 'flag', 'eez', 'illegal']]

seascape_dummies = pd.get_dummies(moddat['seascape_class']).reset_index(drop=True)
month_dummies = pd.get_dummies(moddat['month']).reset_index(drop=True)
flag_dummies = pd.get_dummies(moddat['flag']).reset_index(drop=True)
moddat = pd.concat([moddat, seascape_dummies, month_dummies, flag_dummies], axis=1)


# Get X, y
y = moddat[['fishing_hours']].reset_index(drop=True)
y.loc[:, 'fishing_hours'] = np.log(1 + y.fishing_hours)
moddat = moddat.drop(columns = ['month', 'fishing_hours', 'seascape_class', 'flag'])
X = moddat
X.columns
X.head()
y.head()


