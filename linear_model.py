import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

dat = pd.read_feather('data/full_gfw_effort_model_data_8DAY_2012-01-01_2016-12-26.feather')
dat = dat[dat.geartype == 'drifting_longlines']
dat = dat.sort_values('date')

# ~50% of obs are zero (remove?)
len(dat[dat.fishing_hours > 0])/len(dat)
dat = dat[dat.fishing_hours > 0]

# Remove 2015 because incomplete (still processing)
dat = dat[dat.year <= 2014]

# Linear model

# Get data frame of variables and dummy seascapes
moddat = dat[['fishing_hours', 'month', 'seascape_class', 'sst', 'chlor_a']].dropna().reset_index(drop=True)
seascape_dummies = pd.get_dummies(moddat['seascape_class']).reset_index(drop=True)
month_dummies = pd.get_dummies(moddat['month']).reset_index(drop=True)
moddat = pd.concat([moddat, month_dummies], axis=1)

# Get X, y
y = moddat['fishing_hours'].reset_index(drop=True)
y = np.log(1 + y)
moddat = moddat.drop(columns = ['month', 'fishing_hours', 'seascape_class'])
X = moddat
X.head()
y.head()

mod = sm.OLS(y, X).fit()
mod.summary()

# ----------------------------------
# EDA
eda = dat[['date', 'fishing_hours']]
eda.loc[:, 'month'] = pd.DatetimeIndex(eda['date']).month
eda.loc[:, 'year'] = pd.DatetimeIndex(eda['date']).year

eda.loc[:, 'date'] = eda['month'].astype(str) + '-' + eda['year'].astype(str)
eda.head()

# Plot density of fishing hours
sns.distplot(eda['fishing_hours'])
sns.distplot(np.log(1 + eda['fishing_hours']))


# Plot months
sns.barplot('month', y='fishing_hours', data=eda, color='b', estimator=np.sum)
plt.xticks(rotation=90)

# Plot month-year
sns.barplot('date', y='fishing_hours', data=eda, color='b', estimator=np.sum)
plt.xticks(rotation=90)

# Scatter plot
sns.scatterplot(x='sst', y='fishing_hours', data = dat)

sns.scatterplot(x='chlor_a', y='fishing_hours', data = dat)




# Correlation plot
cordat = moddat
cordat['fishing_hours'] = y
cordat.head()

corr = cordat.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(16, 6))
sns.heatmap(corr, annot=True, mask=mask)

# ----------------------------------





