import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from sklearn.preprocessing import MinMaxScaler
from statsmodels.discrete.discrete_model import Logit
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# files = glob.glob('data/processed/*.feather')
# files
# list_ = []
# for file in files:
#     df = pd.read_feather(file)
#     list_.append(df)
#     mdat = pd.concat(list_, sort=False)


# mdat = mdat.reset_index(drop=True)

# dat = mdat

# Full processed data
dat = pd.read_feather('data/full_gfw_10d_effort_model_data_8DAY_2012-01-01_2016-12-26.feather')

dat = dat[dat.geartype == 'drifting_longlines']
dat = dat.sort_values('date')

# ~50% of obs are zero (remove?)
len(dat[dat.fishing_hours > 0])/len(dat)
dat = dat[dat.fishing_hours > 0]


# Linear model
# Get data frame of variables and dummy seascapes
moddat = dat[['fishing_hours', 'flag', 'year', 'month', 'seascape_class', 'sst', 'sst_grad', 'sst4', 'sst4_grad', 'chlor_a', 'lon1', 'lat1', 'tonnage', 'depth_m', 'coast_dist_km', 'port_dist_km']].dropna().reset_index(drop=True)

# Square sst
moddat['sst_sq'] = moddat['sst']**2
#moddat = moddat[['fishing_hours', 'month', 'seascape_class', 'sst', 'sst_sq', 'chlor_a', 'flag', 'eez', 'illegal']]

seascape_dummies = pd.get_dummies(moddat['seascape_class']).reset_index(drop=True)
month_dummies = pd.get_dummies(moddat['month']).reset_index(drop=True)
flag_dummies = pd.get_dummies(moddat['flag']).reset_index(drop=True)
moddat = pd.concat([moddat, seascape_dummies, month_dummies, flag_dummies], axis=1)


# Get X, y
y = moddat[['fishing_hours', 'year']].reset_index(drop=True)
y.loc[:, 'fishing_hours'] = np.log(1 + y.fishing_hours)
moddat = moddat.drop(columns = ['month', 'fishing_hours', 'seascape_class', 'flag'])
X = moddat
X.columns
X.head()
y.head()

mod = sm.OLS(y, X).fit()
mod.summary()



# Rolling-year Cross validate
pred_score = pd.DataFrame()
for year in range(2013, 2017):
    X_train = moddat[moddat.year < year]
    y_train = y[y.year < year]
    X_test = moddat[moddat.year == year]
    y_test = y[y.year == year]

    y_train = y_train['fishing_hours']
    y_test = y_test['fishing_hours']
    
    # Random Forest
    #mod = RandomForestRegressor().fit(X_train,y_train)
    
    # OLS
    mod = sm.OLS(y_train, X_train).fit()
    y_train_pred = mod.predict(X_train)
    y_test_pred = mod.predict(X_test)


    train_score = mean_squared_error(y_train, y_train_pred)
    test_score = mean_squared_error(y_test, y_test_pred)
    
    rdat = pd.DataFrame({'year': [year], 'train_score': [train_score], 'test_score': [test_score]})
    pred_score = pd.concat([pred_score, rdat])

print(pred_score)




# ----------------------------------
# EDA
eda = dat[['date', 'fishing_hours']]
eda.loc[:, 'month'] = pd.DatetimeIndex(eda['date']).month
eda.loc[:, 'year'] = pd.DatetimeIndex(eda['date']).year

eda.loc[:, 'date'] = eda['month'].astype(str) + '-' + eda['year'].astype(str)
eda.head()

# Plot density of fishing hours
sns.distplot(eda['fishing_hours'])
plt.show()

sns.distplot(np.log(1 + eda['fishing_hours']))
plt.show()

# Plot months
sns.barplot('month', y='fishing_hours', data=eda, color='b', estimator=np.sum)
plt.xticks(rotation=90)
plt.show()

# Plot month-year
sns.barplot('date', y='fishing_hours', data=eda, color='b', estimator=np.sum)
plt.xticks(rotation=90)
plt.show()

# Scatter plot
sns.scatterplot(x='sst', y='fishing_hours', data = dat)
plt.show()

sns.scatterplot(x='chlor_a', y='fishing_hours', data = dat)
plt.show()


# Correlation plot
cordat = moddat
cordat['fishing_hours'] = y
cordat.head()
cordat.corr()

corr = cordat.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(16, 6))
sns.heatmap(corr, annot=True, mask=mask)
plt.show()

# ----------------------------------





