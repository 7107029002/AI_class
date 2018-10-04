import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
import seaborn as sns
from scipy import stats

data = pandas.read_csv('brain_size.csv', sep=';', na_values=".")
t = np.linspace(-6, 6, 20)
sin_t = np.sin(t)
cos_t = np.cos(t)
pandas.DataFrame({'t': t, 'sin': sin_t, 'cos': cos_t})
data.shape
data.columns
print(data['Gender'])
data[data['Gender'] == 'Female']['VIQ'].mean()
groupby_gender = data.groupby('Gender')
for gender, value in groupby_gender['VIQ']:
    print((gender, value.mean()))
groupby_gender.mean()
from pandas.tools import plotting
plotting.scatter_matrix(data[['Weight', 'Height', 'MRI_Count']])
plotting.scatter_matrix(data[['PIQ', 'VIQ', 'FSIQ']])
stats.ttest_1samp(data['VIQ'], 0)
female_viq = data[data['Gender'] == 'Female']['VIQ']
male_viq = data[data['Gender'] == 'Male']['VIQ']
stats.ttest_ind(female_viq, male_viq)
stats.ttest_ind(data['FSIQ'], data['PIQ'])
stats.ttest_rel(data['FSIQ'], data['PIQ'])
stats.ttest_1samp(data['FSIQ'] - data['PIQ'], 0)
stats.wilcoxon(data['FSIQ'], data['PIQ'])
#23結束

x = np.linspace(-5, 5, 20)
np.random.seed(1)
# normal distributed noise
y = -5 + 3*x + 4 * np.random.normal(size=x.shape)
# Create a data frame containing all the relevant variables
data = pandas.DataFrame({'x': x, 'y': y})
from statsmodels.formula.api import ols
model = ols("y ~ x", data).fit()
print(model.summary())

#30
data_fisq = pandas.DataFrame({'iq': data['FSIQ'], 'type': 'fsiq'})
data_piq = pandas.DataFrame({'iq': data['PIQ'], 'type': 'piq'})
data_long = pandas.concat((data_fisq, data_piq))
print(data_long)
model = ols("iq ~ type", data_long).fit()
print(model.summary())


