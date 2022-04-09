from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  # plotting
import numpy as np  # linear algebra
import os  # accessing directory structure
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression  # linear regression
from sklearn.model_selection import train_test_split  # split train and test data
from sklearn.preprocessing import PolynomialFeatures  # polynomial regression
import seaborn as sns
import statsmodels.api as sm
sns.set()

import os


df = pd.read_csv('./data/measurements.csv')
df['distance']=df['distance'].str.replace(',','.').astype(float)
df['consume']=df['consume'].str.replace(',','.').astype(float)
df['temp_inside']=df['temp_inside'].str.replace(',','.').astype(float)
dataFrames = pd.get_dummies(df, columns=['distance', 'consume', 'speed','temp_inside','temp_outside','specials','gas_type','AC','rain','sun','refill liters','refill gas'], drop_first=True)

# df = pd.read_excel('./data/measurements2.xlsx',sheet_name='Sheet1')

df.head()
print(df.head())

df.describe()
print(df.describe())

y=df['consume']
x1=df['distance']

print(y)
print(x1)

plt.scatter(x1,y)
plt.xlabel('Distance',fontsize=20)
plt.ylabel('Consume',fontsize=20)
plt.show()
plt.close()

# x = df[['distance', 'speed', 'temp_inside']]
x = sm.add_constant(x1)
results=sm.OLS(y,x).fit()
results.summary()
print(results.summary())


plt.scatter(x1,y)
yhat = -0.0059*x1 + 5.0279
fig = plt.plot(x1,yhat,lw=4,c='orange',label='regression line')
plt.xlabel('Distance',fontsize=20)
plt.ylabel('Consume',fontsize=20)
plt.show()
