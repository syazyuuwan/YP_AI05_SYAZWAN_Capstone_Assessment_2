#%%
# Import packages and setup

import os, datetime

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

from time_series_helper import WindowGenerator


mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# %%
# define function to compile and fit dataset

learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

MAX_EPOCHS = 20

def compile_and_fit(model, window, model_type, patience=2, epochs=MAX_EPOCHS, ):
  if model_type == 1:
    base_log_path=r"tensorboard_logs\single_step"
  else:
    base_log_path=r"tensorboard_logs\multi_step"
    
  log_path = os.path.join(base_log_path,datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  tb = tf.keras.callbacks.TensorBoard(log_path)
  
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=optimizer,
                metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])

  history = model.fit(window.train, epochs=epochs,
                      validation_data=window.val,
                      callbacks=[early_stopping,tb])
  return history
#%%
# %%
CSV_PATH = os.path.join(os.getcwd(),'cases_malaysia_covid.csv')
# %%
# Load dataset
df = pd.read_csv(CSV_PATH)

# subsetting to required columns only
df = df.copy()[['date','cases_new','cases_import','cases_recovered','cases_active']]
# %%
# Inspecting data
df.info()
#%%
# Convert column to int
df['cases_new'] = pd.to_numeric(df['cases_new'], errors='coerce', downcast='integer').astype('Int64')
# %%
# null values
df.isnull().sum()
#%%
df=df.dropna()
# %%
# Duplicated values
df.duplicated().sum()
# %%
# separate 'date' column
date = pd.to_datetime(df.pop('date'), format='%d/%m/%Y')
#%%
#Plotting dataset to check for trends

plot_cols = ['cases_new']
plot_features = df[plot_cols]
plot_features.index = date
_ = plot_features.plot(subplots=True)

plot_features = df[plot_cols][:480]
plot_features.index = date[:480]
_ = plot_features.plot(subplots=True)
# %%
# split data
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]
#%%
#normalize data

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std
# %%
#inspect distribution of normalisation
df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
df_std['Normalized'] = df_std['Normalized'].astype('float')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=90)
# %%
#Data windowing
single_window = WindowGenerator(
    input_width=30, label_width=30, shift=1,
    label_columns=['cases_new'],
    train_df=train_df, val_df=val_df, test_df=test_df
)
#%%
multi_window = WindowGenerator(
    input_width=30, label_width=30, shift=30,
    label_columns=['cases_new'],
    train_df=train_df, val_df=val_df, test_df=test_df
)
# %%
#single step LSTM model
single_lstm = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dense(units=1)
])
# %%
history = compile_and_fit(single_lstm, single_window, model_type=1, patience=3, epochs=100)
# %%
single_window.plot(plot_col='cases_new',model=single_lstm)
#%%
print('Evaluation for single-step model:\n',single_lstm.evaluate(single_window.test))
# %%
#multi-step LTSM model
#%%
for inputs, labels in multi_window.train.take(1):
  print(f'Inputs shape (batch, time, features): {inputs.shape}')
  print(f'Labels shape (batch, time, features): {labels.shape}')
# %%
multi_lstm = tf.keras.Sequential()
multi_lstm.add(tf.keras.layers.LSTM(128,return_sequences=False))
multi_lstm.add(tf.keras.layers.Dense(30*labels.shape[-1]))
multi_lstm.add(tf.keras.layers.Reshape([30,labels.shape[-1]]))
multi_lstm.add(tf.keras.layers.LSTM(64,return_sequences=False))
multi_lstm.add(tf.keras.layers.Dense(30*labels.shape[-1]))
multi_lstm.add(tf.keras.layers.Reshape([30,labels.shape[-1]]))
# %%
history = compile_and_fit(multi_lstm,multi_window, model_type=2,patience=3, epochs = 50)
# %%
multi_window.plot('cases_new',multi_lstm)
# %%
print('Evaluation for multi-step model:\n',multi_lstm.evaluate(single_window.test))
# %%
tf.keras.utils.plot_model(single_lstm)
# %%

tf.keras.utils.plot_model(multi_lstm)