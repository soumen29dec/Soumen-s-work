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
from sklearn.model_selection import train_test_split

data=pd.read_csv('E:/Internship/Machine Learning/Arbitrage/NSE_2017_EQ_Data.csv')
edel=data[data["SYMBOL"]=="EDELWEISS"]
hdfc=data[data["SYMBOL"]=="HDFCBANK"]
axis=data[data["SYMBOL"]=="AXISBANK"]
icici=data[data["SYMBOL"]=="ICICIBANK"]
kotak=data[data["SYMBOL"]=="KOTAKBANK"]
indiabull=data[data['SYMBOL']=='IBULHSGFIN']

header=["HDFCBANK", 'AXISBANK', 'ICICIBANK', 'KOTAKBANK', 'INDIABULL', 'EDEWEISS']
newData=pd.DataFrame()

#adding 6 stocks and target in new dataframe
newData["HDFCBANK"]=hdfc["CLOSE"].values
newData["AXISBANK"]=axis["CLOSE"].values
newData['ICICIBANK']=icici["CLOSE"].values
newData["KOTAKBANK"]=kotak["CLOSE"].values
newData["INDIABULL"]=indiabull["CLOSE"].values
newData["EDELWEISS"]=edel["CLOSE"].values
print(newData.describe())
print(newData.head())

#TARGET==> EDELWEISS
x=newData.drop('EDELWEISS', axis=1)
x_list=list(x.columns)
y=np.array(newData['EDELWEISS'])
print("Shape of Feature: ", x.shape)
X=np.array(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
X_train, ValData, y_train, ValLabel = train_test_split(X, y, test_size=0.2, random_state=50)
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
y_pred17_edel = model.predict(X_test)
y_pred17_edel=pd.DataFrame(y_pred17_edel)
np.savetxt("E:/Internship/Machine Learning/Arbitrage/2017_pred_edel.csv",y_pred17_edel,delimiter=',')

#Target==> HDFC
x=newData.drop('HDFCBANK', axis=1)
x_list=list(x.columns)
y=np.array(newData['HDFCBANK'])
print("Shape of Feature: ", x.shape)
X=np.array(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
X_train, ValData, y_train, ValLabel = train_test_split(X, y, test_size=0.2, random_state=50)
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
y_pred17_hdfc = model.predict(X_test)
y_pred17_hdfc=pd.DataFrame(y_pred17_hdfc)
print("Size of HDFC Bank's 2017 predicted Stock Price: ", len(y_pred17_hdfc))
np.savetxt("E:/Internship/Machine Learning/Arbitrage/2017_pred_hsfc.csv",y_pred17_hdfc,delimiter=',')

#Target==> AXISBANK
x=newData.drop('AXISBANK', axis=1)
x_list=list(x.columns)
y=np.array(newData['AXISBANK'])
print("Shape of Feature: ", x.shape)
X=np.array(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
X_train, ValData, y_train, ValLabel = train_test_split(X, y, test_size=0.2, random_state=50)
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
y_pred17_axis = model.predict(X_test)
y_pred17_axis=pd.DataFrame(y_pred17_axis)
print("Size of Axis Bank's 2017 Predicted Stock Price: ", len(y_pred17_axis))
np.savetxt("E:/Internship/Machine Learning/Arbitrage/2017_pred_axis.csv",y_pred17_axis,delimiter=',')

#Target==> ICICI Bank

x=newData.drop('ICICIBANK', axis=1)
x_list=list(x.columns)
y=np.array(newData['ICICIBANK'])
print("Shape of Feature: ", x.shape)
X=np.array(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
X_train, ValData, y_train, ValLabel = train_test_split(X, y, test_size=0.2, random_state=50)
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
y_pred17_icici = model.predict(X_test)
y_pred17_icici=pd.DataFrame(y_pred17_icici)
print("Size of ICICI Bank's 2017 Predicted Stock Price: ", len(y_pred17_icici))
np.savetxt("E:/Internship/Machine Learning/Arbitrage/2017_pred_icici.csv",y_pred17_icici,delimiter=',')

#Target==> Kotak Bank
x=newData.drop('KOTAKBANK', axis=1)
x_list=list(x.columns)
y=np.array(newData['KOTAKBANK'])
print("Shape of Feature: ", x.shape)
X=np.array(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
X_train, ValData, y_train, ValLabel = train_test_split(X, y, test_size=0.2, random_state=50)
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
y_pred17_kotak = model.predict(X_test)
y_pred17_kotak=pd.DataFrame(y_pred17_kotak)
print("Size of Kotak Bank's 2017 Predicted Stock Price: ", len(y_pred17_kotak))
np.savetxt("E:/Internship/Machine Learning/Arbitrage/2017_pred_kotak.csv",y_pred17_kotak,delimiter=',')

#Targte==> India Bulls
x=newData.drop('INDIABULL', axis=1)
x_list=list(x.columns)
y=np.array(newData['INDIABULL'])
print("Shape of Feature: ", x.shape)
X=np.array(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
X_train, ValData, y_train, ValLabel = train_test_split(X, y, test_size=0.2, random_state=50)
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
y_pred17_ind = model.predict(X_test)
y_pred17_ind=pd.DataFrame(y_pred17_ind)
print("Size of India Bull's 2017 Predicted Stock Price: ", len(y_pred17_ind))
np.savetxt("E:/Internship/Machine Learning/Arbitrage/2017_pred_indiabull.csv",y_pred17_ind,delimiter=',')

#Creating 2018 Predicted DataSet
data2018 = pd.read_csv("E:/Internship/Machine Learning/Arbitrage/2018_predicted_data.csv", delimiter=',')
predData=pd.DataFrame()
predData['HDFCBANK']=data2018['HDFCBANK'].values
predData['ICICIBANK']=data2018['ICICIBANK'].values
predData['AXISBANK']=data2018['AXISBANK'].values
predData['KOTAKBANK']=data2018['KOTAKBANK'].values
predData['INDIABULLS']=data2018['INDIABULLS'].values
predData['EDELWEISS']=data2018['EDELWEISS'].values
print(predData.describe())
print(predData.head())
x=predData.drop('EDELWEISS', axis=1)
x_list=list(x.columns)
print('Shape of Features: ',x.shape)
X=np.array(x)
y=np.array(predData['EDELWEISS'])

#Using KNN Regression to determine 2018 Predicted Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
X_train, ValData, y_train, ValLabel = train_test_split(X, y, test_size=0.05)
print('Shape of Training Feature: ', X_train.shape)
print('Shape of Testing Features: ', X_test.shape)
print('Shape of Training Labels: ', y_train.shape)
print('Shape of Testing Labels: ', y_test.shape)


kvals=range(1,5,1)
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
y_pred18 = model.predict(X_test)
y_pred18=pd.DataFrame(y_pred18)
print("Size of EDEWEISS's 2018 Predicted Stock Price: ", len(y_pred18))
np.savetxt("E:/Internship/Machine Learning/Arbitrage/2018_pred.csv",y_pred18,delimiter=',')

#Using Linear Regression
reg=LinearRegression()
reg.fit(X_train, y_train)
y_pred18_LG = reg.predict(X_test)
print(pd.DataFrame(y_pred18_LG))
np.savetxt("E:/Internship/Machine Learning/Arbitrage/2018_pred_LG.csv",y_pred18_LG,delimiter=',')
np.savetxt("E:/Internship/Machine Learning/Arbitrage/2017_ytest.csv",y_test,delimiter=',')

#PLOT Testing Data with Predicted Data
plt.figure(figsize=(10,8))
plt.plot(y_test, label='2018 Projected Test Date of Target EDELWEISS')
plt.plot(y_pred18_LG, label='2019 Predicted Price of Target EDELWEISS')
plt.title('Price Comparison of 2018 Sample Prices with 2019 Predicted Prices')
plt.legend(loc=0)
plt.show()
#Correlation Matrix
import seaborn as sns
corr=predData.corr()
sns.heatmap(corr)
#Sort Correlation Coefficient from highest to lowest
print("Correlation Coefficients from highest to lowest with Target: ")
print(corr['EDELWEISS'].sort_values(ascending=False))

#plot 2018 expected price of EDELWEISS
plt.figure(figsize=(10,8))
plt.plot(predData['EDELWEISS'], label='Expected Price Trend of Target Stock in 2018')
plt.title('Expected Price Trend of EDELWEISS Stock in 2018')
plt.show()

plt.figure(figsize=(10,8))
plt.plot(predData['HDFCBANK'], label='Expected Price Trend of HDFC Bank in 2018')
plt.title('Expected Price Trend of HDFC BANK Stock in 2018')
plt.show()

plt.figure(figsize=(10,8))
plt.plot(predData['ICICIBANK'], label='Expected Price Trend of ICICI Bank in 2018')
plt.title('Expected Price Trend of ICICI BANK Stock in 2018')
plt.show()

plt.figure(figsize=(10,8))
plt.plot(predData['AXISBANK'], label='Expected Price Trend of AXIS Bank in 2018')
plt.title('Expected Price Trend of AXIX BANK Stock in 2018')
plt.show()

plt.figure(figsize=(10,8))
plt.plot(predData['KOTAKBANK'], label='Expected Price Trend of KOTAK Bank in 2018')
plt.title('Expected Price Trend of KOTAK Bank Stock in 2018')
plt.show()

plt.figure(figsize=(10,8))
plt.plot(predData['INDIABULLS'], label='Expected Price Trend of India Bulls in 2018')
plt.title('Expected Price Trend of INDIA BULLS Stock in 2018')
plt.show()

def zscore(series):
    return (series - series.mean())/np.std(series)

#Pairing HDFC with EDELWRISS
diff_series = predData['HDFCBANK'].values - predData['EDELWEISS'].values
plt.plot(zscore(pd.DataFrame(diff_series)), label='Arbitrage with HDFC')
plt.title('Z-Score of Target Stock with HDFC Bank')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='green', linestyle='--')
plt.xlabel('Serial Number of Price Data')
plt.ylabel('Z=Score')
plt.show()

#Paring ICICI with EDELWEISS
diff_series = predData['ICICIBANK'].values - predData['EDELWEISS'].values
plt.plot(zscore(pd.DataFrame(diff_series)), label='Arbitrage with HDFC')
plt.title('Z-Score of Target Stock with ICICI Bank')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='green', linestyle='--')
plt.xlabel('Serial Number of Price Data')
plt.ylabel('Z=Score')
plt.show()

diff_series = predData['AXISBANK'].values - predData['EDELWEISS'].values
plt.plot(zscore(pd.DataFrame(diff_series)), label='Arbitrage with AXIS')
plt.title('Z-Score of Target Stock with AXIS Bank')
plt.axhline(1.0, color='blue', linestyle='--')
plt.axhline(-1.0, color='green', linestyle='--')
plt.xlabel('Serial Number of Price Data')
plt.ylabel('Z=Score')
plt.show()

diff_series = predData['KOTAKBANK'].values - predData['EDELWEISS'].values
plt.plot(zscore(pd.DataFrame(diff_series)), label='Arbitrage with KOTAK')
plt.title('Z-Score of Target Stock with KOTAK Bank')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='green', linestyle='--')
plt.xlabel('Serial Number of Price Data')
plt.ylabel('Z=Score')
plt.show()

diff_series = predData['INDIABULLS'].values - predData['EDELWEISS'].values
plt.plot(zscore(pd.DataFrame(diff_series)), label='Arbitrage with KOTAK')
plt.title('Z-Score of Target Stock with INDIA BULLS')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='blue', linestyle='--')
plt.xlabel('Serial Number of Price Data')
plt.ylabel('Z=Score')
plt.show()
