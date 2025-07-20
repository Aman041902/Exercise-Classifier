import pandas as pd
from glob import glob



files = glob("../../data/raw/MetaMotion/*.csv")
len(files)

files[0]

data_path = "../../data/raw/MetaMotion\\"
f=files[0]
f.split("-")[0].replace(data_path, "")

participant = f.split("-")[0].replace(data_path, "")
label = f.split("-")[1]
category = f.split("-")[2].rstrip("123")

df = pd.read_csv(f)

df['participant'] = participant
df['label'] = label
df['category'] = category

files = glob("../../data/raw/MetaMotion/*.csv")

def read_data_files(files):
  acc_df = pd.DataFrame()
  gyro_df = pd.DataFrame()

  acc_set = 1
  gyro_set = 1

  for f in files:
    participant = f.split("-")[0].replace(data_path, "")
    label = f.split("-")[1]
    category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

    df = pd.read_csv(f)

    df['participant'] = participant
    df['label'] = label
    df['category'] = category

    if "Accelerometer" in f:
      df['set'] = acc_set
      acc_set += 1
      acc_df = pd.concat([acc_df, df], axis=0)
    else:
      df['set'] = gyro_set
      gyro_set += 1
      gyro_df = pd.concat([gyro_df, df], axis=0)

  acc_df.index = pd.to_datetime(acc_df['epoch (ms)'], unit='ms')
  gyro_df.index = pd.to_datetime(gyro_df['epoch (ms)'], unit='ms')

  del gyro_df['epoch (ms)']
  del gyro_df['time (01:00)']
  del gyro_df['elapsed (s)']

  del acc_df['epoch (ms)']
  del acc_df['time (01:00)']
  del acc_df['elapsed (s)']

  return acc_df, gyro_df



acc_df, gyro_df = read_data_files(files)

merged_data = pd.concat([acc_df.iloc[:, :3], gyro_df], axis=1)

merged_data.info()

merged_data.columns = ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z","participant","label","category","set"]

sampling = {
  "acc_x":"mean","acc_y":"mean","acc_z":"mean","gyro_x":"mean","gyro_y":"mean","gyro_z":"mean","participant":"last","label":"last","category":"last","set":"last"
}

days = [g for n,g in merged_data.groupby(pd.Grouper(freq='D'))]

len(days)

data_resampled = pd.concat([df.resample('200ms').agg(sampling).dropna() for df in days])

data_resampled['set'] = data_resampled['set'].astype(int)
data_resampled.info()

data_resampled.to_pickle('../../data/interim/data_processed.pkl')

  
