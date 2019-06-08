import numpy as np
import pandas as pd
import pandas_datareader as pdr
from statsmodels.tsa.api import adfuller
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score



data=pd.read_csv('E:/Internship/Machine Learning/Arbitrage/NSE_2017_EQ_Data.csv')
tgt=data[data["SYMBOL"]=="EDELWEISS"]
hdfc=data[data["SYMBOL"]=="HDFCBANK"]
axis=data[data["SYMBOL"]=="AXISBANK"]
icici=data[data["SYMBOL"]=="ICICIBANK"]
kotak=data[data["SYMBOL"]=="KOTAKBANK"]
indiabull=data[data['SYMBOL']=='IBULHSGFIN']
print("Size of Target: ", len(tgt))
print('Size of HDFC Bank Data: ', len(hdfc))
print("Size of Axis Bank Data: ", len(axis))
print("Size of ICICI Bank Data: ", len(icici))
print('Size of Kotak Bank Data: ', len(kotak))
print('Size of India Bull Data: ', len(indiabull))

tgt=data[data["SYMBOL"]=="EDELWEISS"]
tgt['TIMESTAMP'] = tgt['TIMESTAMP'].apply(lambda d: date2num(datetime.datetime.strptime(d, "%m/%d/%Y")))
plt.plot_date(x=tgt['TIMESTAMP'], y=tgt['CLOSE'])
plt.title('Stock Price of EDELWEISS in 2017')
plt.ylabel('Price')
plt.xlabel('Date')
plt.grid(True)
plt.show()

hdfc['TIMESTAMP'] = hdfc['TIMESTAMP'].apply(lambda d: date2num(datetime.datetime.strptime(d, "%m/%d/%Y")))
plt.plot_date(x=hdfc['TIMESTAMP'], y=hdfc['CLOSE'])
plt.title('Stock Price of HDFC BANK in 2017')
plt.ylabel('Price')
plt.xlabel('Date')
plt.grid(True)
plt.show()

axis['TIMESTAMP'] = axis['TIMESTAMP'].apply(lambda d: date2num(datetime.datetime.strptime(d, "%m/%d/%Y")))
plt.plot_date(x=axis['TIMESTAMP'], y=axis['CLOSE'])
plt.title('Stock Price of AXIS BANK in 2017')
plt.ylabel('Price')
plt.xlabel('Date')
plt.grid(True)
plt.show()

kotak['TIMESTAMP'] = kotak['TIMESTAMP'].apply(lambda d: date2num(datetime.datetime.strptime(d, "%m/%d/%Y")))
plt.plot_date(x=kotak['TIMESTAMP'], y=kotak['CLOSE'])
plt.title('Stock Price of KOTAK BANK in 2017')
plt.ylabel('Price')
plt.xlabel('Date')
plt.grid(True)
plt.show()

icici['TIMESTAMP'] = icici['TIMESTAMP'].apply(lambda d: date2num(datetime.datetime.strptime(d, "%m/%d/%Y")))
plt.plot_date(x=icici['TIMESTAMP'], y=icici['CLOSE'])
plt.title('Stock Price of ICICI BANK in 2017')
plt.ylabel('Price')
plt.xlabel('Date')
plt.grid(True)
plt.show()

indiabull['TIMESTAMP'] = indiabull['TIMESTAMP'].apply(lambda d: date2num(datetime.datetime.strptime(d, "%m/%d/%Y")))
plt.plot_date(x=indiabull['TIMESTAMP'], y=indiabull['CLOSE'])
plt.title('Stock Price of INDIA BULLS in 2017')
plt.ylabel('Price')
plt.xlabel('Date')
plt.grid(True)
plt.show()

#Creating new dataframe with component stocks and target stocks
header=["HDFCBANK", 'AXISBANK', 'ICICIBANK', 'KOTAKBANK', 'INDIABULL', 'TARGET']
newData=pd.DataFrame()

#adding 6 stocks and target in new dataframe
newData["HDFCBANK"]=hdfc["CLOSE"].values
newData["AXISBANK"]=axis["CLOSE"].values
newData['ICICIBANK']=icici["CLOSE"].values
newData["KOTAKBANK"]=kotak["CLOSE"].values
newData["INDIABULL"]=indiabull["CLOSE"].values
newData["TARGET"]=tgt["CLOSE"].values
print(newData.describe())
print(newData.head())

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

reg=LinearRegression()
reg.fit(X_train, y_train)
y_pred_2017 = reg.predict(X_test)
print(y_pred_2017)
np.savetxt("E:/Internship/Machine Learning/Arbitrage/2017_prediction.csv",y_pred_2017,delimiter=',')
np.savetxt("E:/Internship/Machine Learning/Arbitrage/2017_test.csv",y_test,delimiter=',')

#Using KNN Model
X_train, ValData, y_train, ValLabel = train_test_split(X, y, test_size=0.1, random_state=50)
kvals=range(1,10,2)
accuracies=[]
for k in kvals:
    model=KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)
    score=model.score(ValData, ValLabel)
    print('k=%d, accuracy=%.2f%%' % (k, score * 100))
    accuracies.append(score)

i=np.argmax(accuracies)
print("k=%d, achieved highest accuracy of %.2f%%" %(kvals[i], accuracies[i]*100))
model=KNeighborsRegressor(n_neighbors=kvals[i])
model.fit(X_train, y_train)
y_pred_17_knn = model.predict(X_test)
y_pred_17_knn=pd.DataFrame(y_pred_17_knn)
np.savetxt("E:/Internship/Machine Learning/Arbitrage/2017_pred_KNN.csv",y_pred_17_knn,delimiter=',')

plt.figure(figsize=(10,8))
plt.plot(y_test, label="2017 Actual Stock Price of EDELWEISS")
plt.plot(y_pred_2017, label="2017 Predicted Stock Price of EDELWEISS")
plt.title('Price Comparison of Actual Target Prices with Predicted Target Prices')
plt.legend(loc=0)
plt.show()

plt.figure(figsize=(10,8))
plt.plot(y_test, label="2017 Actual Stock Price of EDELWEISS")
plt.plot(y_pred_17_knn, label="2017 Predicted Stock Price of EDELWEISS")
plt.title('Price Comparison of Actual Target Prices with Predicted Target Prices using KNN')
plt.legend(loc=0)
plt.show()

#Correlation Matrix
import seaborn as sns
corr=newData.corr()
sns.heatmap(corr)
#Sort Correlation Coefficient from highest to lowest
print("Correlation Coefficients from highest to lowest with Target: ")
print(corr['TARGET'].sort_values(ascending=False))

test_price=pd.read_csv('E:/Internship/Machine Learning/Arbitrage/2016_2017_prediction.csv')
pred_16=test_price['2016_predicted_price']
pred_17=test_price['2017_predicted_price']

plt.figure(figsize=(10,8))
plt.plot(pred_16, label="2016 Predicted Price of EDELWEISS")
plt.plot(pred_17, label="2017 Predicted Price of EDELWEISS")
plt.title('Price Comparison of 2017 Predicted Prices with 2016 Predicted Prices')
plt.legend(loc=0)
plt.show()
#Cointegration between HDFC and Target
result_hdfc = coint(hdfc['CLOSE'], tgt['CLOSE'])
print('Score: ', result_hdfc[0])
print('pValue: ', result_hdfc[1])
diff_series_hdfc = hdfc['CLOSE'].values - tgt['CLOSE'].values
tgt_s = data[data["SYMBOL"]=="EDELWEISS"]
tgt_s['TIMESTAMP'] = tgt_s['TIMESTAMP'].apply(lambda d: date2num(datetime.datetime.strptime(d, '%m/%d/%Y')))
plt.plot_date(x=tgt_s['TIMESTAMP'], y=pd.DataFrame(diff_series_hdfc), label='Price Difference')
#plt.gca().xaxis.set_major_formatter(DateFormatter('%m/%d/%Y'))
plt.title('Price Diffeence between HDFC and Target Stock')
#plt.legend(loc=0)
plt.xlabel('Date')
plt.ylabel('Difference in Price')
plt.grid(True)
plt.show()

#Cointegration between Axis and Target
result_axis = coint(axis['CLOSE'], tgt['CLOSE'])
print('Score: ', result_axis[0])
print('pValue: ', result_axis[1])
diff_series_axis = axis['CLOSE'].values - tgt['CLOSE'].values
plt.plot_date(x=tgt_s['TIMESTAMP'], y=pd.DataFrame(diff_series_axis), label='Price Difference')
#plt.gca().xaxis.set_major_formatter(DateFormatter('%m/%d/%Y'))
plt.title('Price Diffeence between AXIS and Target Stock')
#plt.legend(loc=0)
plt.xlabel('Date')
plt.ylabel('Difference in Price')
plt.grid(True)
plt.show()

#Cointegration between ICICI and Target
result_icici = coint(icici['CLOSE'], tgt['CLOSE'])
print('Score: ', result_icici[0])
print('pValue: ', result_icici[1])
diff_series_icici = icici['CLOSE'].values - tgt['CLOSE'].values
plt.plot_date(x=tgt_s['TIMESTAMP'], y=pd.DataFrame(diff_series_icici), label='Price Difference')
#plt.gca().xaxis.set_major_formatter(DateFormatter('%m/%d/%Y'))
plt.title('Price Diffeence between ICICI and Target Stock')
#plt.legend(loc=0)
plt.xlabel('Date')
plt.ylabel('Difference in Price')
plt.grid(True)
plt.show()

#Cointegration between KOTAK and Target
result_kotak = coint(kotak['CLOSE'], tgt['CLOSE'])
print('Score: ', result_kotak[0])
print('pValue: ', result_kotak[1])
diff_series_kotak = kotak['CLOSE'].values - tgt['CLOSE'].values
plt.plot_date(x=tgt_s['TIMESTAMP'], y=pd.DataFrame(diff_series_kotak), label='Price Difference')
#plt.gca().xaxis.set_major_formatter(DateFormatter('%m/%d/%Y'))
plt.title('Price Diffeence between KOTAK and Target Stock')
#plt.legend(loc=0)
plt.xlabel('Date')
plt.ylabel('Difference in Price')
plt.grid(True)
plt.show()

#Cointegration between KOTAK and Target
result_ib = coint(indiabull['CLOSE'], tgt['CLOSE'])
print('Score: ', result_ib[0])
print('pValue: ', result_ib[1])
diff_series_ib = indiabull['CLOSE'].values - tgt['CLOSE'].values
plt.plot_date(x=tgt_s['TIMESTAMP'], y=pd.DataFrame(diff_series_ib), label='Price Difference')
#plt.gca().xaxis.set_major_formatter(DateFormatter('%m/%d/%Y'))
plt.title('Price Diffeence between INDIA BULLS and Target Stock')
#plt.legend(loc=0)
plt.xlabel('Date')
plt.ylabel('Difference in Price')
plt.grid(True)
plt.show()

def zscore(series):
    return (series - series.mean())/np.std(series)
#zscore(pd.DataFrame(diff_series))
plt.plot_date(x=tgt_s['TIMESTAMP'], y=zscore(pd.DataFrame(diff_series_hdfc)), label='Arbitrage with HDFC')
plt.title('Z-Score of Target Stock with HDFC Bank')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='green', linestyle='--')
#plt.legend(loc=0)
plt.xlabel('Date')
plt.ylabel('Z=Score')
plt.grid(True)
plt.show()

plt.plot_date(x=tgt_s['TIMESTAMP'], y=zscore(pd.DataFrame(diff_series_axis)), label='Arbitrage with AXIS')
plt.title('Z-Score of Target Stock with AXIS Bank')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='blue', linestyle='--')
#plt.legend(loc=0)
plt.xlabel('Date')
plt.ylabel('Z=Score')
plt.grid(True)
plt.show()

diff_series_icici = icici['CLOSE'].values - tgt['CLOSE'].values
plt.plot_date(x=tgt_s['TIMESTAMP'], y=zscore(pd.DataFrame(diff_series_icici)), label='Arbitrage with AXIS')
plt.title('Z-Score of Target Stock with ICICI Bank')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='green', linestyle='--')
#plt.legend(loc=0)
plt.xlabel('Date')
plt.ylabel('Z=Score')
plt.grid(True)
plt.show()

plt.plot_date(x=tgt_s['TIMESTAMP'], y=zscore(pd.DataFrame(diff_series_kotak)), label='Arbitrage with KOTAK')
plt.title('Z-Score of Target Stock with KOTAK Bank')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='blue', linestyle='--')
#plt.legend(loc=0)
plt.xlabel('Date')
plt.ylabel('Z=Score')
plt.grid(True)
plt.show()

plt.plot_date(x=tgt_s['TIMESTAMP'], y=zscore(pd.DataFrame(diff_series_ib)), label='Arbitrage with IB')
plt.title('Z-Score of Target Stock with India Bulls')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='blue', linestyle='--')
#plt.legend(loc=0)
plt.xlabel('Date')
plt.ylabel('Z=Score')
plt.grid(True)
plt.show()

data_16 =pd.read_csv('E:/Internship/Machine Learning/Arbitrage/NSE_2016_EQ_Data.csv')
hdfc_16 = data_16[data_16['SYMBOL']=='HDFCBANK']
tgt_16 = data_16[data_16['SYMBOL']=='EDELWEISS']
axis_16 = data_16[data_16['SYMBOL']=='AXISBANK']
icici_16 = data_16[data_16['SYMBOL']=='ICICIBANK']
kotak_16 = data_16[data_16['SYMBOL']=='KOTAKBANK']
indiabull_16 = data_16[data_16['SYMBOL']=='IBULHSGFIN']

hdfc_17 = hdfc[hdfc.TIMESTAMP !='12/29/2017']
tgt_17 = tgt[tgt.TIMESTAMP !='12/29/2017']
axis_17 = axis[axis.TIMESTAMP !='12/29/2017']
icici_17 = icici[icici.TIMESTAMP !='12/29/2017']
kotak_17 = kotak[kotak.TIMESTAMP !='12/29/2017']
indiabull_17 = indiabull[indiabull.TIMESTAMP !='12/29/2017']

print('length of modified HDFC data: ', len(hdfc_17))
print('length of modified target data: ', len(tgt_17))
print('length of modified axis data: ', len(axis_17))
print('length of modified icici data: ', len(icici_17))
print('length of modified kotak data: ', len(kotak_17))
print('length of modified India Bull data: ', len(indiabull_17))

hdfc_ratio = hdfc_17['CLOSE'].values/hdfc_16['CLOSE'].values
tgt_ratio = tgt_17['CLOSE'].values/tgt_16['CLOSE'].values
axis_ratio = axis_17['CLOSE'].values/axis_16['CLOSE'].values
icici_ratio = icici_17['CLOSE'].values/icici_16['CLOSE'].values
kotak_ratio = kotak_17['CLOSE'].values/kotak_16['CLOSE'].values
ib_ratio = indiabull_17['CLOSE'].values/indiabull_16['CLOSE'].values

#Creating 2018 Projected Dataset using Predicted Stock Price

