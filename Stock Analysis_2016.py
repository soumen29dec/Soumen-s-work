import numpy as np
import pandas as pd
import pandas_datareader as pdr
from statsmodels.tsa.api import adfuller
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
import matplotlib.dates as mdates
import datetime
from datetime import date
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY, date2num


data=pd.read_csv('E:/Internship/Machine Learning/Arbitrage/NSE_2016_EQ_Data.csv')

#get filtered data for TARGET (EDELWEISS), HDFCBANK, AXISBANK, MFSL, ICICIBANK, KOTAKBANK & IBULHSGFIN
tgt=data[data["SYMBOL"]=="EDELWEISS"]
tgt['TIMESTAMP'] = tgt['TIMESTAMP'].apply(lambda d: date2num(datetime.datetime.strptime(d, "%m/%d/%Y")))
plt.plot_date(x=tgt['TIMESTAMP'], y=tgt['CLOSE'])
plt.title('Stock Price of EDELWEISS in 2016')
plt.ylabel('Price')
plt.grid(True)
plt.show()

#bar plot
'''fmt = mdates.DateFormatter('%d/%m/%Y')
loc = mdates.WeekdayLocator(byweekday=mdates.MONDAY)
ax = plt.axes()
ax.xaxis.set_major_formatter(fmt)
ax.xaxis.set_major_locator(loc)
plt.bar(tgt['CLOSE'], tgt['TIMESTAMP'], align="center")
plt.title('Stock Price of EDELWEISS in 2016')
plt.ylabel('Price')
plt.grid(True)
fig=plt.figure(1)
fig.autofmt_xdate()
plt.show()'''

print("Size of Target: ", len(tgt))

hdfc=data[data["SYMBOL"]=="HDFCBANK"]
hdfc=data[data["SYMBOL"]=="HDFCBANK"]
hdfc['TIMESTAMP'] = hdfc['TIMESTAMP'].apply(lambda d: date2num(datetime.datetime.strptime(d, "%m/%d/%Y")))
plt.plot_date(x=hdfc['TIMESTAMP'], y=hdfc['CLOSE'])
plt.title('Stock Price of HDFC Bank in 2016')
plt.ylabel('Price')
plt.grid(True)
plt.show()
print('Size of HDFC Bank Data: ', len(hdfc))

axis=data[data["SYMBOL"]=="AXISBANK"]
axis=data[data["SYMBOL"]=="AXISBANK"]
axis['TIMESTAMP'] = axis['TIMESTAMP'].apply(lambda d: date2num(datetime.datetime.strptime(d, "%m/%d/%Y")))
plt.plot_date(x=axis['TIMESTAMP'], y=axis['CLOSE'])
plt.title('Stock Price of Axis Bank in 2016')
plt.ylabel('Price')
plt.grid(True)
plt.show()
print("Size of Axis Bank Data: ", len(axis))

maxfin=data[data["SYMBOL"]=="MFSL"]
maxfin['TIMESTAMP'] = maxfin['TIMESTAMP'].apply(lambda d: date2num(datetime.datetime.strptime(d, "%m/%d/%Y")))
plt.plot_date(x=maxfin['TIMESTAMP'], y=maxfin['CLOSE'])
plt.title('Stock Price of MAX FINANCIAL in 2016')
plt.ylabel('Price')
plt.xlabel('Time')
plt.grid(True)
plt.show()
print('Size of Max Financial Data: ', len(maxfin))

icici=data[data["SYMBOL"]=="ICICIBANK"]
icici['TIMESTAMP'] = icici['TIMESTAMP'].apply(lambda d: date2num(datetime.datetime.strptime(d, "%m/%d/%Y")))
plt.plot_date(x=icici['TIMESTAMP'], y=icici['CLOSE'])
plt.title('Stock Price of ICICI Bank in 2016')
plt.ylabel('Price')
plt.xlabel('Time')
plt.grid(True)
plt.show()
print('Size of ICICI Bank Data: ', len(icici))

kotak=data[data["SYMBOL"]=="KOTAKBANK"]
kotak['TIMESTAMP'] = kotak['TIMESTAMP'].apply(lambda d: date2num(datetime.datetime.strptime(d, "%m/%d/%Y")))
plt.plot_date(x=kotak['TIMESTAMP'], y=kotak['CLOSE'])
plt.title('Stock Price of KOTAK Bank in 2016')
plt.ylabel('Price')
plt.xlabel('Time')
plt.grid(True)
plt.show()
print('Size of Kotak Bank: ', len(kotak))

indiabull=data[data['SYMBOL']=='IBULHSGFIN']
indiabull['TIMESTAMP'] = indiabull['TIMESTAMP'].apply(lambda d: date2num(datetime.datetime.strptime(d, "%m/%d/%Y")))
plt.plot_date(x=indiabull['TIMESTAMP'], y=indiabull['CLOSE'])
plt.title('Stock Price of INDIA BULLS in 2016')
plt.ylabel('Price')
plt.xlabel('Time')
plt.grid(True)
plt.show()
print('Size of India Bull Finance Data: ', len(indiabull))

#plot Closing price of Target and 7 other stocks

plt.figure(figsize=(10,8))
plt.plot(hdfc["CLOSE"], label="HDFC Bank")
plt.plot(axis["CLOSE"], label="AXIS Bank")
plt.plot(indiabull["CLOSE"], label="India Bull")
#plt.plot(bajaj["CLOSE"], label="Bajaj Finance")
plt.plot(maxfin["CLOSE"], label="Max Financial Svc. Ltd.")
plt.plot(icici["CLOSE"], label="ICICI Bank")
plt.plot(kotak["CLOSE"], label="Kotak Bank")
#plt.plot(sbi["CLOSE"], label="SBI Bank")
plt.plot(tgt["CLOSE"], label = "TARGET")
plt.title('Price movement of Target with Inida Bull and Kotak Bank over 2016')
plt.legend(loc=0)
plt.show()

#Creating new dataframe with component stocks and target stocks
header=["HDFCBANK", 'AXISBANK', 'MAXFIN', 'ICICIBANK', 'KOTAKBANK', 'INDIABULL', 'TARGET']
newData=pd.DataFrame()

#adding 6 stocks and target in new dataframe
newData["HDFCBANK"]=hdfc["CLOSE"].values
newData["AXISBANK"]=axis["CLOSE"].values
newData["MAXFIN"]=maxfin["CLOSE"].values
newData['ICICIBANK']=icici["CLOSE"].values
newData["KOTAKBANK"]=kotak["CLOSE"].values
newData["INDIABULL"]=indiabull["CLOSE"].values
newData["TARGET"]=tgt["CLOSE"].values
print(newData.describe())
print(newData.head())

#Correlation Matrix
import seaborn as sns
corr=newData.corr()
sns.heatmap(corr)
#Sort Correlation Coefficient from highest to lowest
print("Correlation Coefficients from highest to lowest with Target: ")
print(corr['TARGET'].sort_values(ascending=False))

#Creating x and y split
x=newData.drop('MAXFIN', axis=1)
x=newData.drop('TARGET', axis=1)
x_list=list(x.columns)
y=np.array(newData['TARGET'])
print("Shape of Feature: ", x.shape)
X=np.array(x)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print('Training Feature Shape: ', X_train.shape)
print('Training Labels Shape: ', y_train.shape)
print('Testing Feature Shape: ', X_test.shape)
print('Testing Labels Shape: ', y_test.shape)

#Linear Regression
#estimate co-effecient
import sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
reg=LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(y_pred)
np.savetxt("E:/Internship/Machine Learning/Arbitrage/2016_prediction.csv",y_pred,delimiter=',')
np.savetxt("E:/Internship/Machine Learning/Arbitrage/2016_test.csv",y_test,delimiter=',')

#plot comparision of prediction with test data
plt.figure(figsize=(10,8))
plt.plot(y_test, label="2016 Actual Stock Price of EDELWEISS")
plt.plot(y_pred, label="2016 Precicted Stock Price of EDELWEISS")
plt.title('Price Comparison of Actual Target Prices with Predicted Target Prices')
plt.legend(loc=0)
plt.show()


#Evaluating Model
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
ex_var_score = explained_variance_score(y_test, y_pred)
m_absolute_error = mean_absolute_error(y_test, y_pred)
m_squared_error = mean_squared_error(y_test, y_pred)
r_2_score = r2_score(y_test, y_pred)


#Calculating Z-score of pair of stocks

import statsmodels
from statsmodels.tsa.stattools import coint
from datetime import date
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY, date2num
import datetime

#Cointegration between HDFC and Target
result_hdfc = coint(hdfc['CLOSE'], tgt['CLOSE'])
print('Score: ', result_hdfc[0])
print('pValue: ', result_hdfc[1])

diff_series = hdfc['CLOSE'].values - tgt['CLOSE'].values
tgt_s = data[data["SYMBOL"]=="EDELWEISS"]
tgt_s['TIMESTAMP'] = tgt_s['TIMESTAMP'].apply(lambda d: date2num(datetime.datetime.strptime(d, '%m/%d/%Y')))
plt.plot_date(x=tgt_s['TIMESTAMP'], y=pd.DataFrame(diff_series), label='Price Difference')
#plt.gca().xaxis.set_major_formatter(DateFormatter('%m/%d/%Y'))
plt.title('Price Diffeence between HDFC and Target Stock')
#plt.legend(loc=0)
plt.xlabel('Date')
plt.ylabel('Difference in Price')
plt.grid(True)
plt.show()

def zscore(series):
    return (series - series.mean())/np.std(series)
#zscore(pd.DataFrame(diff_series))
plt.plot_date(x=tgt_s['TIMESTAMP'], y=zscore(pd.DataFrame(diff_series)), label='Arbitrage with HDFC')
plt.title('Z-Score of Target Stock with HDFC Bank')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='green', linestyle='--')
#plt.legend(loc=0)
plt.xlabel('Date')
plt.ylabel('Z=Score')
plt.grid(True)
plt.show()

#Co-integration of Axis Bank and Target
result_axis=coint(axis['CLOSE'], tgt['CLOSE'])
print('Score: ', result_axis[0])
print('pValue: ', result_axis[1])

diff_series_axis=axis['CLOSE'].values - tgt['CLOSE'].values
#tgt_s['TIMESTAMP'] = tgt_s['TIMESTAMP'].apply(lambda d: date2num(datetime.datetime.strptime(d, '%m/%d/%Y')))
plt.plot_date(x=tgt_s['TIMESTAMP'], y=pd.DataFrame(diff_series_axis), label='Price Difference')
#plt.gca().xaxis.set_major_formatter(DateFormatter('%m/%d/%Y'))
plt.title('Price Diffeence between AXIS Bank and Target Stock')
#plt.legend(loc=0)
plt.xlabel('Date')
plt.ylabel('Difference in Price')
plt.grid(True)
plt.show()

#plot z-score for axis-target pair
plt.plot_date(x=tgt_s['TIMESTAMP'], y=zscore(pd.DataFrame(diff_series_axis)), label='Arbitrage with Axis Bank')
plt.title('Z-Score of Target Stock with Axis Bank')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='blue', linestyle='--')
#plt.legend(loc=0)
plt.xlabel('Date')
plt.ylabel('Z=Score')
plt.grid(True)
plt.show()

#Co-integration of ICICI and Target Stock
result_icici = coint(icici['CLOSE'], tgt['CLOSE'])
print('Score: ', result_icici[0])
print('pValue: ', result_icici[1])

diff_series_icici=icici['CLOSE'].values - tgt['CLOSE'].values
#tgt_s['TIMESTAMP'] = tgt_s['TIMESTAMP'].apply(lambda d: date2num(datetime.datetime.strptime(d, '%m/%d/%Y')))
plt.plot_date(x=tgt_s['TIMESTAMP'], y=pd.DataFrame(diff_series_icici), label='Price Difference')
#plt.gca().xaxis.set_major_formatter(DateFormatter('%m/%d/%Y'))
plt.title('Price Diffeence between ICICI Bank and Target Stock')
#plt.legend(loc=0)
plt.xlabel('Date')
plt.ylabel('Difference in Price')
plt.grid(True)
plt.show()

#plot z-score for ICICI-target pair
plt.plot_date(x=tgt_s['TIMESTAMP'], y=zscore(pd.DataFrame(diff_series_icici)), label='Arbitrage with Axis Bank')
plt.title('Z-Score of Target Stock with ICICI Bank')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='blue', linestyle='--')
#plt.legend(loc=0)
plt.xlabel('Date')
plt.ylabel('Z=Score')
plt.grid(True)
plt.show()

#Co-integration of Kotak and Target Stock
result_kotak = coint(kotak['CLOSE'], tgt['CLOSE'])
print('Score: ', result_kotak[0])
print('pValue: ', result_kotak[1])

diff_series_kotak=kotak['CLOSE'].values - tgt['CLOSE'].values
#tgt_s['TIMESTAMP'] = tgt_s['TIMESTAMP'].apply(lambda e: date2num(datetime.datetime.strptime(e, '%m/%d/%Y')))
plt.plot_date(x=tgt_s['TIMESTAMP'], y=pd.DataFrame(diff_series_kotak), label='Price Difference')
plt.gca().xaxis.set_major_formatter(DateFormatter('%m/%d/%Y'))
plt.title('Price Difference between Kotak Bank and Target Stock')
#plt.legend(loc=0)
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.show()

#plot z-score for Kotak-target pair
plt.plot_date(x=tgt_s['TIMESTAMP'], y=zscore(pd.DataFrame(diff_series_kotak)), label='Arbitrage with Axis Bank')
plt.title('Z-Score of Target Stock with KOTAK Bank')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='blue', linestyle='--')
#plt.legend(loc=0)
plt.xlabel('Date')
plt.ylabel('Z=Score')
plt.grid(True)
plt.show()

#Co-integration of India Bulls and Target Stock
result_ib = coint(indiabull['CLOSE'], tgt['CLOSE'])
print('Score: ', result_ib[0])
print('pValue: ', result_ib[1])

diff_series_ib=indiabull['CLOSE'].values - tgt['CLOSE'].values
#indiabull_s['TIMESTAMP'] = indiabull_s['TIMESTAMP'].apply(lambda e: date2num(datetime.datetime.strptime(e, '%m/%d/%Y')))
plt.plot_date(x=tgt_s['TIMESTAMP'], y=pd.DataFrame(diff_series_ib), label='Price Difference')
plt.gca().xaxis.set_major_formatter(DateFormatter('%m/%d/%Y'))
plt.title('Price Difference between India Bulls Fin and Target Stock')
#plt.legend(loc=0)
plt.xlabel('Date')
plt.ylabel('Difference in Price')
plt.grid(True)
plt.show()
#plot z-score for India Bulls-target pair
plt.plot_date(x=tgt_s['TIMESTAMP'], y=zscore(pd.DataFrame(diff_series_ib)), label='Arbitrage with India Bulls Bank')
plt.title('Z-Score of Target Stock with India Bulls Bank')
plt.axhline(1.0, color='green', linestyle='--')
plt.axhline(-1.0, color='blue', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Z=Score')
plt.show()


# 2017 STOCK ANALYSIS, PREDICTION, ARBITRAGE OPPORTUNITIES
#----------------------------------------------------------------------

