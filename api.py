import random
from bokeh.models import (HoverTool, FactorRange, Plot, LinearAxis, Grid,
                          Range1d)
from bokeh.models.glyphs import VBar
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models.sources import ColumnDataSource
from flask import Flask, render_template
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from numpy import split
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import StringIO
import base64
from pandas import DataFrame
from pandas import concat


app = Flask(__name__)

def LinearRegressionfunction(df,offset):
    df= df.drop(['TotalReactivePower','CurrentIntensity','Device_1','Device_2','Device_3','Voltage'],axis=1)
    
    X,y = to_supervised(df,offset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # models=()

    model=LinearRegression()
    # img = StringIO.StringIO()
    
    # fig, ax = pyplot.subplots(nrows=2, ncols=2, figsize=(14, 14))
    #i=0
   
   
    # for row in ax:
    #     for col in row:
            # model=models[name[i]]
    nrmse,y_pred = sklearn_predict(model, X_train, y_train,X_test,y_test)

    result=pd.DataFrame()
    result['datetime']=df.datetime.iloc[len(X_train)+offset:]
    #print(len(result.datetime))
    result['predicted']=y_pred
    #print(len(result.predicted))
    result['actual']=y_test
    result['datetime'] = pd.to_datetime(result['datetime'])
    source = ColumnDataSource(result)
    p = figure(x_axis_type="datetime", height=350, width=400)
    p.line(x='datetime', y='predicted', line_width=2, source=source)
    p.line(x='datetime', y='actual',color='red', line_width=2, source=source)
    script, div = components(p)
            # result=pd.DataFrame()
            # result['datetime']=df.datetime.iloc[len(X_train)+offset:]
            # result['predicted']=y_pred
            # result['actual']=y_test
            # rmse = sqrt(mean_squared_error(y_test, y_pred))

            # col.plot(result.datetime[:100],result.predicted[:100],color='blue',label='Predicted')
            # col.xaxis.set_major_locator(pyplot.MaxNLocator(7))
            # col.plot(result.datetime[:100],result.actual[:100],color='red', label='Actual')
            # col.set_xlabel('Date',fontsize=14)
            # col.set_ylabel('Energy Consumption(KW)',fontsize=14)
            # col.set_title(name[i]+' with NRMSE of : '+str(nrmse),fontsize=14)
            # col.legend()
            
    #         print('NRMSE for ',name[i] ,'regression is : ',nrmse)
    #         i=i+1
    # pyplot.savefig(img, format='png')
    # pyplot.close()
    # img.seek(0)

    # plot_url = base64.b64encode(img.getvalue())
    compare=pd.DataFrame({'Actual':y_test,'Predicted ':y_pred})
    return script,div










@app.route("/")
def home():
    return render_template('chart.html')

@app.route("/appliances")
def appliances():
    return render_template('appliances.html')

@app.route("/day")
def day():
    df = pd.read_csv('/home/dell/Documents/Btech Project/ESE Review/SwimmingPoolDays.csv', delimiter=',') 
    script, div=LinearRegressionfunction(df,30)
    return render_template('day.html',script=script, div=div)

@app.route("/month")
def month():
    df = pd.read_csv('/home/dell/Documents/Btech Project/ESE Review/SwimmingPoolMonths.csv', delimiter=',')
    script, div=LinearRegressionfunction(df,12)
    return render_template('month.html',script=script, div=div)


@app.route("/hour",methods=['GET'])
def hour():
    df = pd.read_csv('/home/dell/Documents/Btech Project/ESE Review/SwimmingPool_Hours.csv', delimiter=',') 
    # AR3_model = sm.tsa.AR(df.TotalActivePower).fit(maxlag=24)
    # trialarima(df)
    # plot_url_ar=plot_ar_model(df, AR3_model, 24)
    # plot_url=LinearRegressionfunction(df,24)
    script, div=LinearRegressionfunction(df,24)
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

def to_supervised(df,offset):
    # convert history to a univariate series
    data = df.TotalActivePower
    X, y = list(), list()
    ix_start = 0
    # step over the entire history one time step at a time
    for i in range(len(data)):
        # define the end of the input sequence
        ix_end = ix_start + offset
        # ensure we have enough data for this instance
        if ix_end < len(data):
            X.append(data[ix_start:ix_end])
            y.append(data[ix_end])
        # move along one time step
        ix_start += 1
    return X, y
# prepare a list of ml models
def get_models(models=dict()):
    # linear models
    models['lr'] = LinearRegression()
    # models['lasso'] = Lasso()
    # models['ridge'] = Ridge()
    print('Defined %d models' % len(models))
    return models
# prepare a list of ml models
def getmodels(models=dict()):
    # linear models
    name=['Linear Regression','Lasso Regression','Ridge Regression','SVR']
    models['Linear Regression'] = LinearRegression()
    models['Lasso Regression'] = Lasso()
    models['Ridge Regression'] = Ridge()
    models['SVR'] = SVR(kernel='rbf',epsilon=0.1,gamma='scale')
    print('Defined %d models' % len(models))
    return name,models
# create a feature preparation pipeline for a model
def make_pipeline(model):
    steps = list()
    # standardization
    steps.append(('standardize', StandardScaler()))
    # normalization
    steps.append(('normalize', MinMaxScaler()))
    # the model
    steps.append(('model', model))
    # create pipeline
    pipeline = Pipeline(steps=steps)
    return pipeline
# fit a model and make a forecast
def sklearn_predict(model, X_train, y_train,X_test,y_test):
    
    pipeline = make_pipeline(model)
    # fit the model
    pipeline.fit(X_train,y_train)
    # predict the week, recursively
    y_pred= pipeline.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    score= rmse / (max(y_test) - min(y_test))
    
    return score,y_pred

#function to plot AR time series model with actual usage
# def plot_ar_model(data, model, ar_value):
#     data= data.drop(['TotalReactivePower','CurrentIntensity','Device_1','Device_2','Device_3','Voltage'],axis=1)
#     data.set_index('datetime')
#     data['datetime'] = pd.to_datetime(data['datetime'], format='%Y/%m/%d %H:%M:%S')
#     x = data.datetime
#     y_true = data.TotalActivePower

#     #plot actual usage
#     pyplot.subplots(1,1,figsize=(14,6))
#     pyplot.plot(x,y_true, label='Actual')

#     x_pred = x[ar_value:]
#     y_pred = model.predict()

#     #plot model prediction with AR
#     pyplot.plot(x_pred,y_pred, color='red', label='AR')
    
#     rmse= math.sqrt(mean_squared_error(y_pred, y_true[ar_value:]))

#     pyplot.title("Auto Rrgression with RMSE of -{}".format(rmse))
#     pyplot.xlabel("Date-Time", fontsize=16)
#     pyplot.ylabel("Energy Consumption (kW)", fontsize=16)
#     pyplot.legend()
#     pyplot.savefig(img, format='png')
#     pyplot.close()
#     img.seek(0)

#     plot_url_ar = base64.b64encode(img.getvalue())
#     compare=pd.DataFrame({'Actual':y_test,'Predicted ':y_pred})
#     return plot_url_ar


 




if __name__ == "__main__":
    app.run(debug=True,port="5002")