import random
from bokeh.models import (HoverTool, FactorRange, Plot, LinearAxis, Grid, Range1d)
from bokeh.models.glyphs import VBar
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models.sources import ColumnDataSource
from flask import Flask, render_template
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
# import matplotlib
# matplotlib.use('Agg')
# from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import pandas as pd
from numpy import nan
from numpy import isnan
# import matplotlib.pyplot as plt
import itertools
import warnings


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,r2_score

import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD 
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras import losses
import tensorflow as tf


import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pandas.api.types import is_numeric_dtype
import base64
from pandas import DataFrame
from pandas import concat


app = Flask(__name__)

# def lstmPrediction(df, offset):
#     df = df.set_index('datetime')

#     df = remove_outlier(df)

#     values = df.values
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled = scaler.fit_transform(values)
#     train_sup = series_to_supervised(scaled, 1, 1)
#     train_sup.drop(train_sup.columns[[8,9,10,11,12,13]], axis=1, inplace=True)
    
#     train_X, train_y, test_X, test_y = TrainTestSplit(df, offset)

#     model = Sequential()           
#     model.add(LSTM(80, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
#     model.add(Dropout(0.15))
#     model.add(LSTM(50))
#     model.add(Dropout(0.15))
#     model.add(Dense(1))
#     model.compile(loss=losses.mean_squared_error, optimizer='adam')

#     history = model.fit(train_X, train_y, epochs=25, batch_size=60, validation_data=(test_X, test_y),verbose=2, shuffle=False)

        
#     yhat = model.predict(test_X)
#     test_X = test_X.reshape((test_X.shape[0], 7))

#     # invert scaling for forecast
#     inv_yhat = np.concatenate((yhat, test_X[:, -6:]), axis=1)
#     inv_yhat = scaler.inverse_transform(inv_yhat)
#     inv_yhat = inv_yhat[:,0]

#     # invert scaling for actual
#     test_y = test_y.reshape((len(test_y), 1))
#     inv_y = np.concatenate((test_y, test_X[:, -6:]), axis=1)
#     inv_y = scaler.inverse_transform(inv_y)
#     inv_y = inv_y[:,0]

#     # calculate RMSE
#     rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
#     print('Test RMSE: %.3f' % rmse)


#     return rmse

# def TrainTestSplit(df,offset):
#     values = df.values
#     n_train_time = offset

#     train = values[:n_train_time, :]
#     test = values[n_train_time:, :]

#     # split into input and outputs
#     train_X, train_y = train[:, :-1], train[:, -1]
#     test_X, test_y = test[:, :-1], test[:, -1]

#     # reshape input to be 3D [samples, timesteps, features]
#     train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
#     test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
#     return train_X, train_y, test_X, test_y

# def series_to_supervised(data, lag=1, lead=1, dropnan=True):
#     '''
#         an auxillary function to prepare the dataset with given lag and lead using pandas shift function.
#     '''
#     n_vars = data.shape[1]
#     dff = pd.DataFrame(data)
#     cols, names = [],[]
    
#     for i in range(lag, 0, -1):
#         cols.append(dff.shift(i))
#         names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

#     for i in range(0, lead):
#         cols.append(dff.shift(-i))
#         if i == 0:
#             names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
#         else:
#             names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

#     total = pd.concat(cols, axis=1)
#     total.columns = names
#     if dropnan:
#         total.dropna(inplace=True)
#     return total

# def remove_outlier(df_in):
#     for col_name in df_in.columns:
#         q1 = df_in[col_name].quantile(0.25)
#         q3 = df_in[col_name].quantile(0.75)
#         iqr = q3-q1
#         fence_low  = q1-1.5*iqr
#         fence_high = q3+1.5*iqr
#         df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
#     return df_out





def get_lstm_model(train_X):
    model = Sequential()           
    model.add(LSTM(80, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
    model.add(Dropout(0.15))
    model.add(LSTM(50))
    model.add(Dropout(0.15))
    model.add(Dense(1))
    model.compile(loss=losses.mean_squared_error, optimizer='adam')
    return model

def lstmPrediction(df, look_back):


    # df = df.drop(['Rest_active_power'],axis=1)
    # df = df.set_index('datetime')
    # df.index = pd.to_datetime(df.index)
    df.info()
    data = df.TotalActivePower.values
    data = data.astype('float32')

    # reshaping
    data = np.reshape(data, (-1, 1))
    minmax = MinMaxScaler(feature_range=(0, 1))

    # need to scale the data on training set, then transform the unseen data using this
    # otherwise the model would overfit and achieve better results, which is not ideal for
    # real world use of the model
    train_size = int(len(data) * 0.80)
    test_size = len(data) - train_size
    # train_test_split
    train, test = data[0:train_size,:], data[train_size:len(data),:]
    # scaling
    train = minmax.fit_transform(train)
    test = minmax.transform(test)
    # reshaping into X=t and y=t+1
    # look_back = 100
    X_train, y_train = create_dataset(train, look_back)
    X_test, y_test = create_dataset(test, look_back)

    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # reshaping input into [samples, time_steps, features] format
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    model = Sequential()
    model.add(LSTM(100, return_sequences=True,
                input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    history = model.fit(X_train, y_train, epochs=20, batch_size=256, validation_data=(X_test, y_test), verbose=1, shuffle=False)
  
    # test_predict = model.predict(X_test)
    # # invert predictions
    # test_predict = minmax.inverse_transform(test_predict)
    # y_test = minmax.inverse_transform([y_test])
    
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    # invert predictions
    train_predict = minmax.inverse_transform(train_predict)
    y_train = minmax.inverse_transform([y_train])
    test_predict = minmax.inverse_transform(test_predict)
    y_test = minmax.inverse_transform([y_test])
    
    # print(len( df.iloc[train_size:]))
    # print(len(test_predict))
    # print(len(y_test))
    # print(test_predict)
    
    result=pd.DataFrame()
    result['datetime']= df.datetime.iloc[train_size+look_back+1:]
    # print(len(result.datetime))
    result['predicted']=test_predict
    # print(len(result.predicted))
    result['actual']=y_test[0][:]
    print(len(result.actual))
    result['datetime'] = pd.to_datetime(result['datetime'])
    source = ColumnDataSource(result)
    p = figure(x_axis_type="datetime", height=350, width=400)
    p.line(x='datetime', y='predicted', line_width=2, source=source)
    p.line(x='datetime', y='actual',color='red', line_width=2, source=source)
    script, div = components(p)
    return script,div

# need to create an array of values into a dataset matrix

def create_dataset(dataset, look_back=1):
    X, y = list(), list()
    for i in range(len(dataset)-look_back-1):            
        a = (dataset[i:(i+look_back), 0])
        X.append(a)
        y.append(dataset[i+look_back, 0])
    return np.array(X), np.array(y)

def TrainTestSplit(df):
    # converting to float32 as it takes up less memory and operations can be faster 
    data = df.TotalActivePower.values
    data = data.astype('float32')

    # reshaping
    data = np.reshape(data, (-1, 1))
    minmax = MinMaxScaler(feature_range=(0, 1))

    # need to scale the data on training set, then transform the unseen data using this
    # otherwise the model would overfit and achieve better results, which is not ideal for
    # real world use of the model
    train_size = int(len(data) * 0.80)
    test_size = len(data) - train_size
    # train_test_split
    train, test = data[0:train_size,:], data[train_size:len(data),:]
    # scaling
    train = minmax.fit_transform(train)
    test = minmax.transform(test)
    return train,test



@app.route("/")
def home():

    return render_template('chart.html')

@app.route("/appliances")
def appliances():

    return render_template('appliances.html')

@app.route("/day")
def day():
    df = pd.read_csv('/home/dell/Documents/Btech Project/ESE Review/SwimmingPoolDays.csv', delimiter=',') 
    look_back = 100
    script, div=lstmPrediction(df, look_back)
    return render_template('day.html',script=script, div=div)

@app.route("/month")
def month():
    df = pd.read_csv('/home/dell/Documents/Btech Project/ESE Review/SwimmingPoolMonths.csv', delimiter=',')
    look_back = 100
    script, div=lstmPrediction(df, look_back)
    return render_template('month.html',script=script, div=div)


@app.route("/hour",methods=['GET'])
def hour():
    df = pd.read_csv('/home/dell/Documents/Btech Project/ESE Review/SwimmingPool_Hours.csv', delimiter=',') 
    look_back = 100
    script, div=lstmPrediction(df, look_back)
    return render_template('hour.html',script=script, div=div)   

@app.route("/dayapp")
def dayapp():
    
    return render_template('chart.html')

@app.route("/monthapp")
def monthapp():
   
    return render_template('chart.html')


@app.route("/hourapp",methods=['GET'])
def hourapp():
    
    return render_template('chart.html')   
























# def to_supervised(df,offset):
#     # convert history to a univariate series
#     data = df.TotalActivePower
#     X, y = list(), list()
#     ix_start = 0
#     # step over the entire history one time step at a time
#     for i in range(len(data)):
#         # define the end of the input sequence
#         ix_end = ix_start + offset
#         # ensure we have enough data for this instance
#         if ix_end < len(data):
#             X.append(data[ix_start:ix_end])
#             y.append(data[ix_end])
#         # move along one time step
#         ix_start += 1
#     return X, y
# # prepare a list of ml models
# def get_models(models=dict()):
#     # linear models
#     models['lr'] = LinearRegression()
#     # models['lasso'] = Lasso()
#     # models['ridge'] = Ridge()
#     print('Defined %d models' % len(models))
#     return models
# # prepare a list of ml models
# def getmodels(models=dict()):
#     # linear models
#     name=['Linear Regression','Lasso Regression','Ridge Regression','SVR']
#     models['Linear Regression'] = LinearRegression()
#     models['Lasso Regression'] = Lasso()
#     models['Ridge Regression'] = Ridge()
#     models['SVR'] = SVR(kernel='rbf',epsilon=0.1,gamma='scale')
#     print('Defined %d models' % len(models))
#     return name,models
# # create a feature preparation pipeline for a model
# def make_pipeline(model):
#     steps = list()
#     # standardization
#     steps.append(('standardize', StandardScaler()))
#     # normalization
#     steps.append(('normalize', MinMaxScaler()))
#     # the model
#     steps.append(('model', model))
#     # create pipeline
#     pipeline = Pipeline(steps=steps)
#     return pipeline
# # fit a model and make a forecast
# def sklearn_predict(model, X_train, y_train,X_test,y_test):
    
#     pipeline = make_pipeline(model)
#     # fit the model
#     pipeline.fit(X_train,y_train)
#     # predict the week, recursively
#     rmse = sqrt(mean_squared_error(y_test, y_pred))
#     score= rmse / (max(y_test) - min(y_test))
    
#     return score,y_pred

# #function to plot AR time series model with actual usage
# # def plot_ar_model(data, model, ar_value):
# #     data= data.drop(['TotalReactivePower','CurrentIntensity','Device_1','Device_2','Device_3','Voltage'],axis=1)
# #     data.set_index('datetime')
# #     data['datetime'] = pd.to_datetime(data['datetime'], format='%Y/%m/%d %H:%M:%S')
# #     x = data.datetime
# #     y_true = data.TotalActivePower

# #     #plot actual usage
# #     pyplot.subplots(1,1,figsize=(14,6))
# #     pyplot.plot(x,y_true, label='Actual')

# #     x_pred = x[ar_value:]
# #     y_pred = model.predict()

# #     #plot model prediction with AR
# #     pyplot.plot(x_pred,y_pred, color='red', label='AR')
    
# #     rmse= math.sqrt(mean_squared_error(y_pred, y_true[ar_value:]))

# #     pyplot.title("Auto Rrgression with RMSE of -{}".format(rmse))
# #     pyplot.xlabel("Date-Time", fontsize=16)
# #     pyplot.ylabel("Energy Consumption (kW)", fontsize=16)
# #     pyplot.legend()
# #     pyplot.savefig(img, format='png')
# #     pyplot.close()
# #     img.seek(0)

# #     plot_url_ar = base64.b64encode(img.getvalue())
# #     compare=pd.DataFrame({'Actual':y_test,'Predicted ':y_pred})
# #     return plot_url_ar


# def LinearRegressionFunc(df):
#     df= df.drop(['TotalReactivePower','CurrentIntensity','Device_1','Device_2','Device_3','Voltage'],axis=1)
    
#     X,y = to_supervised(df,offset)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
#     # models=()

#     model=LinearRegression()
#     # img = StringIO.StringIO()
    
#     # fig, ax = pyplot.subplots(nrows=2, ncols=2, figsize=(14, 14))
#     #i=0
   
   
#     # for row in ax:
#     #     for col in row:
#             # model=models[name[i]]
#     nrmse,y_pred = sklearn_predict(model, X_train, y_train,X_test,y_test)

#     result=pd.DataFrame()
#     result['datetime']=df.datetime.iloc[len(X_train)+offset:]
#     #print(len(result.datetime))
#     result['predicted']=y_pred
#     #print(len(result.predicted))
#     result['actual']=y_test
#     result['datetime'] = pd.to_datetime(result['datetime'])
#     source = ColumnDataSource(result)
#     p = figure(x_axis_type="datetime", height=350, width=400)
#     p.line(x='datetime', y='predicted', line_width=2, source=source)
#     p.line(x='datetime', y='actual',color='red', line_width=2, source=source)
#     script, div = components(p)
#             # result=pd.DataFrame()
#             # result['datetime']=df.datetime.iloc[len(X_train)+offset:]
#             # result['predicted']=y_pred
#             # result['actual']=y_test
#             # rmse = sqrt(mean_squared_error(y_test, y_pred))

#             # col.plot(result.datetime[:100],result.predicted[:100],color='blue',label='Predicted')
#             # col.xaxis.set_major_locator(pyplot.MaxNLocator(7))
#             # col.plot(result.datetime[:100],result.actual[:100],color='red', label='Actual')
#             # col.set_xlabel('Date',fontsize=14)
#             # col.set_ylabel('Energy Consumption(KW)',fontsize=14)
#             # col.set_title(name[i]+' with NRMSE of : '+str(nrmse),fontsize=14)
#             # col.legend()
            
#     #         print('NRMSE for ',name[i] ,'regression is : ',nrmse)
#     #         i=i+1
#     # pyplot.savefig(img, format='png')
#     # pyplot.close()
#     # img.seek(0)

#     # plot_url = base64.b64encode(img.getvalue())
#     compare=pd.DataFrame({'Actual':y_test,'Predicted ':y_pred})
#     return script,div


 




if __name__ == "__main__":
    app.run(debug=True,port="5002",threaded=False)
