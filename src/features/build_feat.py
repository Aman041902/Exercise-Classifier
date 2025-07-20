import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../../src/features")
from DataTransformation import LowPassFilter,PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans


df = pd.read_pickle("../../data/interim/data_outliers_removed.pkl")
df.info()

predictor_col = list(df.columns[:6])

for col in predictor_col:
  df[col] = df[col].interpolate()

for s in df['set'].unique():
  duration = df[df['set']==s].index[-1]-df[df['set']==s].index[0]
  df.loc[(df['set']==s),'duration'] = duration.seconds


df.groupby(['category'])['duration'].mean()

df[df['set']==1].index[-1]-df[df['set']==1].index[0]

for s in df['set'].unique() :
  duration = df[df['set']==s].index[-1]-df[df['set']==s].index[0]
  df.loc[(df['set']==s),'duration'] = duration.seconds

df_lowpass = df.copy()
lowpass = LowPassFilter()

df_lowpass.drop('duration',axis=1,inplace=True)
df_lowpass.info()

fs = 1000/200

cutoff = 1.2
df_lowpass = lowpass.low_pass_filter(df_lowpass,'acc_x',fs,cutoff,order=5)

for col in predictor_col:
  df_lowpass = lowpass.low_pass_filter(df_lowpass,col,fs,cutoff,order=5)
  df_lowpass[col] = df_lowpass[col+'_lowpass']
  del df_lowpass[col+"_lowpass"]

df_pca = df_lowpass.copy()
pca = PrincipalComponentAnalysis()
df_pca = pca.apply_pca(df_pca,predictor_col,3)
df_pca.info()

df_squared = df_pca.copy()
acc_r = df_squared['acc_x']**2 + df_squared['acc_y']**2 + df_squared['acc_z']**2
gyro_r = df_squared['gyro_x']**2 + df_squared['gyro_y']**2 + df_squared['gyro_z']**2



df_squared['acc_r'] = np.sqrt(acc_r)
df_squared['gyro_r'] = np.sqrt(gyro_r)

Numabs = NumericalAbstraction()

predictor_col = predictor_col+ ['acc_r', 'gyro_r']
df_temporal = df_squared.copy()

ws = int(1000/200)

for col in predictor_col:
  df_temporal = Numabs.abstract_numerical(df_temporal,[col],ws,'mean')
  df_temporal = Numabs.abstract_numerical(df_temporal,[col],ws,'std')

df_temporal.info()
# df_temporal.drop('duration',axis=1,inplace=True)
df_temporal.info()

temporal_list = []
for s in df_temporal['set'].unique():
  subset = df_temporal[df_temporal['set']==s].copy()
  for col in predictor_col:
    subset = Numabs.abstract_numerical(subset,[col],ws,'mean')
    subset = Numabs.abstract_numerical(subset,[col],ws,'std')

  temporal_list.append(subset)

pd.concat(temporal_list)
df_temporal = pd.concat(temporal_list)


df_freq = df_temporal.copy().reset_index()
freqAbs = FourierTransformation()

fs = int(1000/200)
ws = int(2800/200)

df_freq = freqAbs.abstract_frequency(df_freq,['acc_y'],ws,fs)

df_freq.info()

df_freq_list = []

for s in df_freq['set'].unique():
  subset = df_freq[df_freq['set']==s].reset_index(drop=True).copy()
  subset = freqAbs.abstract_frequency(subset,predictor_col,ws,fs)

  df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list).set_index('epoch (ms)',drop=True)
df_freq = df_freq.dropna()

df_freq = df_freq.iloc[::2]

df_cluster = df_freq.copy()

cluster_col = ['acc_x', 'acc_y', 'acc_z']

k_val = range(2,10)
inertias = []

for k in k_val:
  subset = df_cluster[cluster_col]
  kmeans = KMeans(n_clusters=k,n_init=20, random_state=0)
  cluster_labels = kmeans.fit_predict(subset)
  inertias.append(kmeans.inertia_)


plt.plot(k_val, inertias)
plt.xlabel('k')
plt.ylabel('Inertia')
plt.show()


kmeans = KMeans(n_clusters=5,n_init=20, random_state=0)
subset = df_cluster[cluster_col]
df_cluster['cluster'] = kmeans.fit_predict(subset)

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection='3d')
for c in df_cluster['cluster'].unique():
  subset = df_cluster[df_cluster['cluster']==c]
  ax.scatter(subset['acc_x'], subset['acc_y'], label=c)
plt.legend()
plt.show()

df_cluster.to_pickle('../../data/interim/cluster_data.pkl')