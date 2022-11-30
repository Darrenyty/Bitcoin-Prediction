# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 11:18:43 2022

@author: 123
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
import math
import statsmodels.api as sm
import datetime as dt
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from arch import arch_model

pip install tensorflow

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM

import matplotlib.pyplot as plt
from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

## Read the weekly database
df = pd.read_csv('Daily Database.csv', parse_dates=['Date'], index_col='Date')  # load the data and we call the data as "df"
df_notna=df.dropna()
df.shape
close = df["Adj Close"]

# Run adfuller test for close
result_volume = adfuller(close)
print('ADF Statistic: %f' % result_volume[0])
print('p-value: %f' % result_volume[1])
print('Critical Values:')
for key, value in result_volume[4].items():
    print('\t%s: %.3f' % (key,value))
    
## Differentiate close & adfuller test   
close_diff = close.diff(1).dropna()
result_close_diff = adfuller(close_diff)
print('ADF Statistic: %f' % result_close_diff[0])
print('p-value: %f' % result_close_diff[1])
print('Critical Values:')
for key, value in result_close_diff[4].items():
    print('\t%s: %.3f' % (key,value))
    
## Separate the train and test data
close_diff.shape
m = 183
train_r = close_diff[0:len(close_diff)-m]
test_r = close_diff[len(close_diff)-m:len(close_diff)]

## For loop to find the best ARMA
p = 11
q = 11
mae = np.zeros((p,q))
for i in range(p):
    for j in range(q):
        model = ARIMA(train_r, order=(i,0,j))
        model.initialize_approximate_diffuse()
        model_fit = model.fit()
        forecasted_r = model_fit.forecast(m)
        mae[i,j]=mean_absolute_error(test_r,forecasted_r)
ARIMA_min=np.amin(mae)
ARIMA_min
np.where(mae==np.amin(mae))
mae[8][6] ##906.9174346616373

## Visualize the forecasting
model = ARIMA(train_r, order=(8,0,6))  # Best model is ARMA(8,6)
model.initialize_approximate_diffuse()
model_fit = model.fit()
forecasted_r = model_fit.forecast(10)

x1 = pd.Timestamp('2022-05-16')
x2 = pd.Timestamp('2022-07-18')

plt.figure(figsize=(20,6))
plt.plot(close_diff,linestyle=':', marker='o',color='blue',label = "$gdp_t$")
plt.plot(range(x1,x2), forecasted_r, marker='o',color='black',label="forecasted $gdp_t$")
plt.title("Time paths of $gdp_{t}$ and forecasted $gdp_{t}$")
plt.xlabel("Time")
plt.ylabel("")
plt.legend()
plt.show()
    
## Error Correction Model
depend_variables= df_notna.iloc[: , 1:]
depend_variables.dtypes
depend_variables = depend_variables.apply(pd.to_numeric)


train_depend = depend_variables[0:math.floor(len(df_notna["Gold"])*0.9)]
X = train_depend  # X is a matrix of independent variables
X = sm.add_constant(X) # adding a constant
Y = df_notna['Adj Close'][0:math.floor(len(df_notna["Gold"])*0.9)]

model1 = sm.OLS(Y, X.astype(float))
results1 = model1.fit()
z = results1.resid
z.name = 'residuals'
lag_z = z.shift(1).dropna() 
print(results1.summary())        

X=X.drop(pd.Timestamp('2017-07-21 00:00:00'))
XX = pd.concat([X, lag_z], axis=1).dropna() # put two variables into a matrix X
XX.dtypes
XX = sm.add_constant(XX) # adding a constant
Y=Y.drop(pd.Timestamp('2017-07-21 00:00:00'))
YY = Y        # y is a vector of dependent variable
model2 = sm.OLS(YY, XX)
result2 = model2.fit()
print(result2.summary())

test_dependent = df_notna[(math.floor(len(df_notna["Gold"])*0.9)):math.floor(len(df_notna["Gold"]))]
test_r_notna = test_dependent.iloc[: , 0]
X_test = test_dependent.iloc[: , 1:]  # X is a matrix of independent variables
X_test = sm.add_constant(X_test) # adding a constant
OLS_r=results1.predict(X_test)
mean_absolute_error(test_r_notna,OLS_r) ## 465.53489011294363


residual_test = test_r_notna - OLS_r
lag_residual_test = residual_test.shift(1).dropna() 
X_test=X_test.drop(pd.Timestamp('2022-01-18 00:00:00'))
XX = pd.concat([X_test, lag_residual_test], axis=1).dropna()
XX = sm.add_constant(XX)
ERRORC_r=result2.predict(XX)
test_r_notna=test_r_notna.drop(pd.Timestamp('2022-01-18 00:00:00'))
mean_absolute_error(test_r_notna,ERRORC_r) ## 443.7119274867722


## GARCH
p = 5
q = 5
mae_GARCH = np.zeros((p,q))
for i in range(1,5):
    for j in range(1,5):
        model = arch_model(train_r, mean='AR',lags=1, vol='GARCH', p=i, q=j)
        model_fit = model.fit()
        forecasted_r =  model_fit.forecast(horizon=m)
        forecasted_values = forecasted_r.mean.values[-1, :]
        forecasted_values
        mae_GARCH[i,j] =mean_absolute_error(test_r,forecasted_values)
GARCH_min=np.amin(mae_GARCH)
GARCH_min
np.where(mae_GARCH==np.amin(mae_GARCH))
mae[8][6]

## KNN
df = pd.read_csv('Daily Database.csv')
closedf = df[['Date','Adj Close']]
print("Shape of close dataframe:", closedf.shape)


del closedf["Date"]
scaler=MinMaxScaler(feature_range=(0,1))
closedf=scaler.fit_transform(np.array(closedf).reshape(-1,1))
print(closedf.shape)

training_size=int(len(closedf)*0.90)
test_size=len(closedf)-training_size
train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:1]
print("train_data: ", train_data.shape)
print("test_data: ", test_data.shape)

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 15
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test", y_test.shape)

X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)


## LSTM
df = pd.read_csv('Daily Database.csv')
closedf = df[['Date','Adj Close']]
print("Shape of close dataframe:", closedf.shape)


del closedf["Date"]
scaler=MinMaxScaler(feature_range=(0,1))
closedf=scaler.fit_transform(np.array(closedf).reshape(-1,1))
print(closedf.shape)

training_size=int(len(closedf)*0.90)
test_size=len(closedf)-training_size
train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:1]
print("train_data: ", train_data.shape)
print("test_data: ", test_data.shape)

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 15
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test", y_test.shape)

X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)

model=Sequential()
model.add(LSTM(10,input_shape=(None,1),activation="relu"))
model.add(Dense(1))
model.compile(loss="mean_squared_error",optimizer="adam")

history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=200,batch_size=32,verbose=1)

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()


plt.show()

### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)
train_predict.shape, test_predict.shape

# Transform back to original form

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 

# Evaluation metrices RMSE and MAE
print("Train data RMSE: ", math.sqrt(mean_squared_error(original_ytrain,train_predict)))
print("Train data MSE: ", mean_squared_error(original_ytrain,train_predict))
print("Train data MAE: ", mean_absolute_error(original_ytrain,train_predict))
print("-------------------------------------------------------------------------------------")
print("Test data RMSE: ", math.sqrt(mean_squared_error(original_ytest,test_predict)))
print("Test data MSE: ", mean_squared_error(original_ytest,test_predict))
print("Test data MAE: ", mean_absolute_error(original_ytest,test_predict))

# shift train predictions for plotting
import plotly.io as pio
pio.renderers.default='browser'

look_back=time_step
trainPredictPlot = np.empty_like(closedf)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
print("Train predicted data: ", trainPredictPlot.shape)

# shift test predictions for plotting
testPredictPlot = np.empty_like(closedf)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict
print("Test predicted data: ", testPredictPlot.shape)

names = cycle(['Original close price','Train predicted close price','Test predicted close price'])


plotdf = pd.DataFrame({'date': df['Date'],
                       'original_close': df['Adj Close'],
                      'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                      'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})

fig = px.line(plotdf,x=plotdf['date'], y=[plotdf['original_close'],plotdf['train_predicted_close'],
                                          plotdf['test_predicted_close']],
              labels={'value':'Stock price','date': 'Date'})
fig.update_layout(title_text='Comparision between original close price vs predicted close price',
                  plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
fig.for_each_trace(lambda t:  t.update(name = next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()

## Random Forest 
df = pd.read_csv('Daily Database.csv')
closedf = df[['Date','Adj Close']]
print("Shape of close dataframe:", closedf.shape)

# Normalizing
del closedf['Date']
scaler=MinMaxScaler(feature_range=(0,1))
closedf=scaler.fit_transform(np.array(closedf).reshape(-1,1))
print(closedf.shape)

# Dataset split
training_size=int(len(closedf)*0.90)
test_size=len(closedf)-training_size
train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:1]
print("train_data: ", train_data.shape)
print("test_data: ", test_data.shape)

# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 15
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test", y_test.shape)

from xgboost import XGBRegressor
my_model = XGBRegressor(n_estimators=1000)
my_model.fit(X_train, y_train, verbose=False)

predictions = my_model.predict(X_test)
print("Mean Absolute Error - MAE : " + str(mean_absolute_error(y_test, predictions)))
print("Root Mean squared Error - RMSE : " + str(math.sqrt(mean_squared_error(y_test, predictions))))
train_predict=my_model.predict(X_train)
test_predict=my_model.predict(X_test)
train_predict = train_predict.reshape(-1,1)
test_predict = test_predict.reshape(-1,1)
print("Train data prediction:", train_predict.shape)
print("Test data prediction:", test_predict.shape)
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 
print("Mean Absolute Error - MAE : " + str(mean_absolute_error(original_ytest, test_predict)))
print("Root Mean squared Error - RMSE : " + str(math.sqrt(mean_squared_error(original_ytest, test_predict))))



# shift train predictions for plotting
pio.renderers.default='browser'

look_back=time_step
trainPredictPlot = np.empty_like(closedf)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
print("Train predicted data: ", trainPredictPlot.shape)

# shift test predictions for plotting
testPredictPlot = np.empty_like(closedf)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict
print("Test predicted data: ", testPredictPlot.shape)

names = cycle(['Original close price','Train predicted close price','Test predicted close price'])

plotdf = pd.DataFrame({'date': df['Date'],
                       'original_close': df['Adj Close'],
                      'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                      'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})

fig = px.line(plotdf,x=plotdf['date'], y=[plotdf['original_close'],plotdf['train_predicted_close'],
                                          plotdf['test_predicted_close']],
              labels={'value':'Close price','date': 'Date'})
fig.update_layout(title_text='Comparision between original close price vs predicted close price',
                  plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')
fig.for_each_trace(lambda t:  t.update(name = next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()