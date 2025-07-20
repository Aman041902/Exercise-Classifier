import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display
df = pd.read_pickle("../../data/interim/data_processed.pkl")

labels = df['label'].unique()
participants = df['participant'].unique()

for label in labels:
  for participant in participants:
    all_axis_df = df.query(f"label=='{label}'").query(f"participant=='{participant}'").reset_index()

    if len(all_axis_df) == 0:
      continue
    all_axis_df[['acc_x', 'acc_y', 'acc_z']].plot()
    plt.title(f"{label} - {participant}".title())
    plt.legend()
    plt.show()


for label in labels:
  for participant in participants:
    all_axis_df = df.query(f"label=='{label}'").query(f"participant=='{participant}'").reset_index()

    if len(all_axis_df) > 0:
      all_axis_df[['gyro_x', 'gyro_y', 'gyro_z']].plot()
      plt.title(f"{label} - {participant}".title())
      plt.legend()
      plt.show()

label = "row"
participant = 'A'
combined_plot = (
  df.query(f"label=='{label}'").query(f"participant=='{participant}'").reset_index(drop=True)
  
)

fig,ax = plt.subplots(nrows=2,figsize=(20,10))

combined_plot[['acc_x', 'acc_y', 'acc_z']].plot(ax=ax[0])
combined_plot[['gyro_x', 'gyro_y', 'gyro_z']].plot(ax=ax[1])

ax[1].set_xlabel("samples")
ax[0].set_title(f"{label} - {participant}".title())

for label in labels:
  for participant in participants:
    combined_plot = df.query(f"label=='{label}'").query(f"participant=='{participant}'").reset_index()

    if len(combined_plot) > 0:
      fig,ax = plt.subplots(nrows=2,figsize=(20,10))

      combined_plot[['acc_x', 'acc_y', 'acc_z']].plot(ax=ax[0])
      combined_plot[['gyro_x', 'gyro_y', 'gyro_z']].plot(ax=ax[1])

      ax[1].set_xlabel("samples")
      ax[0].set_title(f"{label} - {participant}".title())
      plt.savefig(f"../../reports/figures/{label}_{participant}.png")

      plt.show()
    
    
