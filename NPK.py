import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_iris
# Code to read csv file into Colaboratory:
!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
from __future__ import print_function
import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
%tensorflow_version 1.x
import tensorflow as tf
from tensorflow.python.data import Dataset
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from fastai.vision import *
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
bs=128
# Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
link = 'https://drive.google.com/open?id=1kBA2Swlty7cK5nOOAAEXAtZ45SqacxzQ'
fluff, id = link.split('=')
print (id) # Verify that you have everything after '='
downloaded = drive.CreateFile({'id':id})
downloaded.GetContentFile('File.csv')
df3 = pd.read_csv('File.csv')
df =df3

df.drop(["S", "SurveyNo","farmername","Mn","Zn","Fe","Cu","Latitude","Textbox50","sample_n"
df.drop
df.dropna(subset = ["EC"], inplace=True)
df.dropna(subset = ["N"], inplace=True)
df.dropna(subset = ["P"], inplace=True)
df.dropna(subset = ["K"], inplace=True)
df.dropna(subset = ["pH"], inplace=True)
df.dropna(subset = ["OC"], inplace=True)
print("THe Data is Loaded")
df['EC'] = df['EC'].astype(str).map(lambda x: x.lstrip('+-').rstrip(' ITCN'))
df['K'] = df['K'].astype(str).map(lambda x: x.lstrip('+-').rstrip(' VLMH'))
df['N'] = df['N'].astype(str).map(lambda x: x.lstrip('+-').rstrip(' VLMH'))
df['pH'] = df['pH'].astype(str).map(lambda x: x.lstrip('+-').rstrip(' MAlscSHNr '))
df['P'] = df['P'].astype(str).map(lambda x: x.lstrip('+-').rstrip(' VLMH'))
df['OC'] = df['OC'].astype(str).map(lambda x: x.lstrip('+-').rstrip(' VLMH'))
df.drop(df.index[0],inplace = True)
df['EC'] = df['EC'].astype(float)
df['K'] = df['K'].astype(float)
df['N'] = df['N'].astype(float)
df['pH'] = df['pH'].astype(float)
df['P'] = df['P'].astype(float)
df['OC'] = df['OC'].astype(float)
X = df[['EC', 'pH','OC']]
y = df[['N','P','K']]
print("Input and Output Variables are seperated..")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
Following code cell trains the linear regressor using the training data.
regressor = LinearRegression(fit_intercept=False,normalize =True)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
df_pred = pd.DataFrame(data=y_pred, columns=["N", "P","K"])
df_test = pd.DataFrame(data = y_test,columns=["N", "P","K"])

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
y = X_test['pH']
x = X_test['EC']
z = y_test['K']
z1 = y_test['N']
z2 = y_test['P']
ax.set_xlabel('EC')
ax.set_ylabel('pH')
ax.set_zlabel('N,P,K')
#ax.set_ybound(lower = 0, upper =14)
#ax.set_xlim(left=0,right=14)
ax.set_ylim(bottom=0,top =14)
#ax.set_ylim(left=0,right=14)
ax.scatter(x, y, z1, c='r', marker='o', label='blue')
ax.scatter(x, y, z, c='b', marker='o', label='blue')
ax.scatter(x, y, z2, c='g', marker='o', label='blue')
plt.title("NPK with EC and pH")
plt.tight_layout()
plt.show()




