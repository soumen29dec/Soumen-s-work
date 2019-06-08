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

data=pd.read_csv('E:/Internship/Machine Learning/Arbitrage/NSE_2016_EQ_Data.csv')
#get filtered data for TARGET (EDELWEISS), HDFCBANK, AXISBANK, ICICIBANK, KOTAKBANK & IBULHSGFIN
tgt=data[data["SYMBOL"]=="EDELWEISS"]
hdfc=data[data["SYMBOL"]=="HDFCBANK"]
print('Size of HDFC Bank Data: ', len(hdfc))

axis=data[data["SYMBOL"]=="AXISBANK"]
print("Size of Axis Bank Data: ", len(axis))

icici=data[data["SYMBOL"]=="ICICIBANK"]
print('Size of ICICI Bank Data: ', len(icici))

kotak=data[data["SYMBOL"]=="KOTAKBANK"]
print('Size of Kotak Bank: ', len(kotak))

indiabull=data[data['SYMBOL']=='IBULHSGFIN']
print('Size of India Bull Finance Data: ', len(indiabull))

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
#Creating x and y split
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
reg=LinearRegression()
reg.fit(X_train, y_train)
y_pred_LReg_2016 = reg.predict(X_test)
print(y_pred_LReg_2016)
np.savetxt("E:/Internship/Machine Learning/Arbitrage/2016_prediction.csv",y_pred_LReg_2016,delimiter=',')
#Evaluating Model
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
ex_var_score = explained_variance_score(y_test, y_pred_LReg_2016)
m_absolute_error = mean_absolute_error(y_test, y_pred_LReg_2016)
m_squared_error = mean_squared_error(y_test, y_pred_LReg_2016)
r_2_score = r2_score(y_test, y_pred_LReg_2016)
print('Variance: ', ex_var_score)
print('Absolute Error:', m_absolute_error)
print('Mean Squared Error:', m_squared_error)
print('R2 Score: ', r_2_score)

#KNN Classifier
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
y_pred_16_knn = model.predict(X_test)
y_pred_16_knn=pd.DataFrame(y_pred_16_knn)
#print("EVALUATION OF TESTING DATA:")
#print(classification_report(y_test, y_pred_16_knn))
#print("Predicted Value:", y_pred_16_knn, "with Probability: ", model.predict_proba(X_test))
np.savetxt("E:/Internship/Machine Learning/Arbitrage/2016_prediction.csv",y_pred_16_knn,delimiter=',')

#plot comparision of prediction with test data
plt.figure(figsize=(10,8))
plt.plot(y_test, label="2016 Actual Stock Price of EDELWEISS")
plt.plot(y_pred_16_knn, label="2016 Predicted Stock Price of EDELWEISS")
plt.title('Price Comparison of Actual Target with Predicted Target Using KNN')
plt.legend(loc=0)
plt.show()


