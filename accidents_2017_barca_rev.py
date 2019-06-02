# -*- coding: utf-8 -*-
"""
Created on Sat May 18 15:48:10 2019

@author: Soumen Sarkar
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import warnings
warnings.filterwarnings("ignore")
import io
import plotly.offline as py#visualization
py.init_notebook_mode(connected=True)#visulatization
import plotly.graph_objs as go#visualization
import plotly.tools as tls#visualization
import plotly.figure_factory as ff

accident=pd.read_csv('D:/Kaggle-DataSet/Barcelona-Dataset/accidents_2017/accidents_2017.csv')
accident.head()
#accident.columns=[col.replace(' ', '_').lower() for col in accident.columns]
accident.columns=[col.replace(' ', '_') for col in accident.columns]

print("Rows     :" ,accident.shape[0])
print("Columns    :" ,accident.shape[1])
print("\nFeatures :  \n", accident.columns.tolist())
print("\nMissing Values:  ", accident.isnull().sum().values.sum())
print("\nUnique Values:   \n", accident.nunique())

df_jan=accident[accident.Month=='January']
df_feb=accident[accident.Month=='February']
df_mar=accident[accident.Month=='March']
df_apr=accident[accident.Month=='April']
df_may=accident[accident.Month=='May']
df_jun=accident[accident.Month=='June']
df_jul=accident[accident.Month=='July']
df_aug=accident[accident.Month=='August']
df_sep=accident[accident.Month=='September']
df_oct=accident[accident.Month=='October']
df_nov=accident[accident.Month=='November']
df_dec=accident[accident.Month=='December']

ID_col = ["Id"]
cat_cols = accident.nunique()[accident.nunique() < 6].keys().tolist()
#cat_cols = [x for x in cat_cols if x not in target_col]
num_cols = [x for x in accident.columns if x not in cat_cols+ID_col]
lab = accident['Month'].value_counts().keys().tolist()
val = accident['Month'].value_counts().values.tolist()

trace = go.Pie(labels=lab,
               values=val,
              marker = dict(colors=[ 'royalblue', 'lime'],
                            line = dict(color='white',
                                        width=1.3)
                            ),
              rotation=90,
              hoverinfo="label+value+text")
layout=go.Layout(dict(title="Accidents by Month",
                      plot_bgcolor="rgb(243,243,243)",
                      paper_bgcolor="rgb(243,243,243)",
                      )
                )
                        
data=[trace]
fig = go.Figure(data=data, layout = layout)
py.iplot(fig, filename="Basic Pie Chart")

target_col=["Month"]
cat_cols_jan=df_dec.nunique()[df_dec.nunique()<6].keys().tolist()
cat_cols_jan=[x for x in cat_cols_jan if x not in target_col]
num_cols_jan = [x for x in df_dec.columns if x not in cat_cols_jan+ID_col+target_col]


def plot_pie(column):
    trace=go.Pie(values=df_dec[column].value_counts().values.tolist(),
                  labels=df_dec[column].value_counts().keys().tolist(),
                  #hoeverinfo="label+percent+name",
                  #name="Accident by Months",
                  domain=dict(x=[0,.48]),
                  marker = dict(line=dict(width=2, color="rgb(243,243,243)")),
                                                                hole=.6)
    
    layout=go.Layout(dict(title="Distribution of Accidents by" +" "+ column,
                          plot_bgcolor="rgb(243,243,243)",
                          paper_bgcolor="rgb(243,243,243)",
                          annotations=[dict(text="December Accidents",
                                            font=dict(size=13),
                                            showarrow=False,
                                            x=.15, y=.5),
                                       
                                       ]
                          )
                    )
    data=[trace]
    fig=go.Figure(data=data, layout=layout)
    py.iplot(fig)
for i in cat_cols_jan:
    plot_pie(i)
    
def histogram(column):
    trace=go.Histogram(x=df_dec[column],
                       histnorm="percent",
                       name="Accident in December",
                       marker=dict(line=dict(width=0.5,color="black",)),
                       opacity=0.9)
    data=[trace]
    layout=go.Layout(dict(title="Distirbution of December Accidents by"+" "+column,
                            plot_bgcolor = "rgb(243,243,243)",
                            paper_bgcolor = "rgb(243,243,243)",
                            xaxis=dict(gridcolor = 'rgb(255,255,255)',
                                       title = column,
                                       zerolinewidth=1,
                                       ticklen=5,
                                       gridwidth=2
                                       ),
                            yaxis = dict(gridcolor = 'rgb(255,255,255)',
                                         title = "percent",
                                         zerolinewidth = 1,
                                         ticklen = 5,
                                         gridwidth = 2
                                         ),
                            ),
                        )
    fig=go.Figure(data=data, layout=layout)
    py.iplot(fig)
for i in num_cols_jan:
    histogram(i)

#determine coefficients between features
    
header=['Id','Mild_injuries','Serious_injuries', 'Victims', 'Vehicles_involved',
        'Longitude','Latitude']
new_df=pd.DataFrame()
new_df['Mild_injuries']=accident['Mild_injuries'].values
new_df['Serious_injuries']=accident['Serious_injuries'].values
new_df['Victims']=accident['Victims'].values
new_df['Vehicles_involved']=accident['Vehicles_involved'].values
new_df['Longitude']=accident['Longitude'].values
new_df['Latitude']=accident['Latitude'].values

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
Id_col = ["Id"]
#Target Columns
target_col = ["Victims"]

def plot_month_scatter(month_group, color):
    tracer = go.Scatter(x = accident[accident["Month"]==month_group]["Victims"],
                        y = accident[accident["Month"]==month_group]["Vehicles_involved"],
                        mode = "markers", marker = dict(line = dict(color = "black",
                                                                    width = .2),
                            size = 4, color = color,
                            symbol = "diamond-dot",
                            ),
                            name = month_group,
                            opacity = .9,
                            )
    return tracer
trace1 = plot_month_scatter("January","#FF3300")
trace2 = plot_month_scatter("February", "#6666FF")
trace3 = plot_month_scatter("March", "#99FF00")
trace4 = plot_month_scatter("April", "#996600")
trace5 = plot_month_scatter("May", "grey")
trace6 = plot_month_scatter("June","purple")
trace7 = plot_month_scatter("July", "brown")
trace8 = plot_month_scatter("August", "yellow")
trace9 = plot_month_scatter("September", "orange")
trace10 = plot_month_scatter("October", "red")
trace11= plot_month_scatter("November", "green")
trace12= plot_month_scatter("December", "blue")

data=[trace1,trace2,trace3,trace4,trace5,trace6,trace7,trace8,trace9,trace10,trace11,trace12]
def layout_title(title):
    layout = go.Layout(dict(title = title,
                            plot_bgcolor = 'rgb(243,243,243)',
                            paper_bgcolor = 'rgb(243,243,243)',
                            xaxis=dict(gridcolor='rgb(255,255,255)',
                                       title = "# Victims",
                                       zerolinewidth=1, ticklen=5, gridwidth=2),
                            yaxis=dict(gridcolor='rgb(255,255,255)',
                                       title="# Vehicles Involved",
                                       zerolinewidth=1, ticklen=5, gridwidth=2),
                            height=600
                            )
                      )
    return layout
layout = layout_title("No. of Victims & Vehicles involved by Months")
#layout2 = layout_title("Monthly Charges & Total Charges by Churn Group")
fig = go.Figure(data=data, layout=layout)
#fig2 = go.Figure(data=data2, layout=layout2)
py.iplot(fig)
#py.iplot(fig2)

avg_acc=accident.groupby(["Month"])[['Victims','Vehicles_involved']].mean().reset_index()

def mean_charges(column):
    tracer = go.Bar(x = avg_acc["Month"],
                    y = avg_acc[column],
                    marker = dict(line = dict(width = 1)),
                    )
    return tracer
def layout_plot(title, xaxis_lab, yaxis_lab):
    layout = go.Layout(dict(title = title,
                            plot_bgcolor = "rgb(243,243,243)",
                            paper_bgcolor = "rgb(243,243,243)",
                            xaxis = dict(gridcolor = "rgb(255,255,255)", title=xaxis_lab,
                                         zerolinewidth=1, ticklen=5, gridwidth=2),
                            yaxis = dict(gridcolor = "rgb(255,255,255)", title=yaxis_lab,
                                         zerolinewidth=1, ticklen=5, gridwidth=2),
                                         ))
    return layout
trace1 = mean_charges("Victims")
layout1 = layout_plot("Average No of Victims by Month",
                      "Month", '# Victims')
data1 = [trace1]
fig1 = go.Figure(data=data1, layout=layout1)

trace2 = mean_charges("Vehicles_involved")
layout2 = layout_plot("Average No of Vechicles by Month",
                      "Month", '# Vechicles')
data2 = [trace2]
fig2 = go.Figure(data=data2, layout=layout2)

py.iplot(fig1)
py.iplot(fig2)

#RUN IT FROM THIS POINT EVERYTIME YOU START SYSTEM FOR PREDICTIONS USING DIFFERENT REGRESSIONS 
accident=pd.read_csv('D:/Kaggle-DataSet/Barcelona-Dataset/accidents_2017/accidents_2017.csv')
df_acc=accident.copy()
df_acc.columns=[col.replace(' ', '_') for col in df_acc.columns]

import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
Id_col=["Id"]
target_col=['Victims']
cat_cols=df_acc.nunique()[df_acc.nunique()<6].keys().tolist()
cat_cols=[x for x in cat_cols if x not in target_col]
num_cols = [x for x in df_acc.columns if x not in cat_cols+target_col+Id_col]
bin_cols = df_acc.nunique()[df_acc.nunique()==2].keys().tolist()
multi_cols = [s for s in cat_cols if s not in bin_cols]
le = LabelEncoder()
for i in bin_cols:
    df_acc[i] = le.fit_transform(df_acc[i])
df_acc=pd.get_dummies(data=df_acc, columns=multi_cols)

std=StandardScaler()
num_cols_scaled=num_cols[5:]
scaled=std.fit_transform(df_acc[num_cols_scaled])
scaled=pd.DataFrame(scaled, columns=num_cols_scaled)
#scaled=std.fit_transform(accident[num_cols])
#scaled=pd.DataFrame(scaled, columns=num_cols)
                 
df_acc=df_acc.drop(columns=num_cols_scaled, axis=1)
df_acc=df_acc.merge(scaled, left_index=True, right_index=True, how='left')
Id_col=['Id']
summary=(df_acc[[i for i in df_acc.columns if i not in Id_col]].
         describe().transpose().reset_index())
summary=summary.rename(columns={"index":"feature"})
summary=np.around(summary,3)
val_lst=[summary['feature'], summary['count'],
         summary['mean'], summary['std'],
         summary['min'], summary['25%'],
         summary['50%'], summary['75%'], summary['max']]
trace=go.Table(header=dict(values=summary.columns.tolist(),
                           line=dict(color=['#506784']),
                                            fill=dict(color=['#119DFF']),
                                                             ),
               cells=dict(values=val_lst,
                          line=dict(color=['#506784']),
                                           fill=dict(color=["lightgrey",'#119DFF']),
                                                            ),
               columnwidth=[200,60,100,100,60,60,80,80,80])
layout=go.Layout(dict(title="Variable Summary"))
figure=go.Figure(data=[trace],layout=layout)
py.iplot(figure)

correlation=df_acc.corr()
matrix_cols=correlation.columns.tolist()
corr_array=np.array(correlation)

trace=go.Heatmap(z=corr_array,
                 x=matrix_cols,
                 y=matrix_cols,
                 colorscale='Viridis',
                 colorbar=dict(title="Pearson Correlation Coefficient",
                               titleside='right'),
                               )
layout=go.Layout(dict(title="Correlation Matrix for variables",
                      autosize=False,
                      height=720,
                      width=800,
                      margin=dict(r=0, l=210,
                                  t=25, b=210),
                      yaxis=dict(tickfont=dict(size=9)),
                      xaxis=dict(tickfont=dict(size=9))))
fig=go.Figure(data=[trace], layout=layout)
py.iplot(fig)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, scorer
from sklearn.metrics import f1_score
import statsmodels.api as sm
from sklearn.metrics import precision_score, recall_score
from yellowbrick.classifier import DiscriminationThreshold
import sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score



id_col=['Id']
target_col=['Victims']
cols=[i for i in df_acc.columns if i not in Id_col + target_col]
cols=cols[5:]
x=df_acc[cols]
X=np.array(x)
y=df_acc[target_col]
train_x,test_x,train_y,test_y = train_test_split(X,y,test_size=0.2,random_state=100)

reg=LinearRegression()
reg.fit(train_x, train_y)
y_pred = reg.predict(test_x)
print(y_pred)
np.savetxt("D:/Kaggle-DataSet/Barcelona-Dataset/accidents_2017/victim_prediction.csv",y_pred,delimiter=',')
np.savetxt("D:/Kaggle-DataSet/Barcelona-Dataset/accidents_2017/test_data.csv",y_pred,delimiter=',')

train_x, ValData, train_y, ValLabel = train_test_split(X, y, test_size=0.2, random_state=100)
kvals=range(1,40,2)
accuracies=[]
for k in kvals:
    model=KNeighborsRegressor(n_neighbors=k)
    model.fit(train_x, train_y)
    score=model.score(ValData, ValLabel)
    print('k=%d, accuracy=%.2f%%' % (k, score * 100))
    accuracies.append(score)

i=np.argmax(accuracies)
print("k=%d, achieved highest accuracy of %.2f%%" %(kvals[i], accuracies[i]*100))
KNN=KNeighborsRegressor(n_neighbors=kvals[i])
KNN.fit(train_x, train_y)
y_pred_knn = KNN.predict(test_x)
y_pred_knn=pd.DataFrame(y_pred_knn)
np.savetxt("D:/Kaggle-DataSet/Barcelona-Dataset/accidents_2017/2017_pred_KNN.csv",y_pred_knn,delimiter=',')

plt.rc("font", size = 14)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
'''logreg = LogisticRegression()
logreg.fit(train_x, train_y)
y_pred = logreg.predict(test_x)
print('Accuracy of logistic regression classifier on test set: {:2f}'.format(logreg.score(test_x,test_y)))

from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold=model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring='accuracy'
results = model_selection.cross_val_score(modelCV, train_x, train_y, cv=kfold, scoring=scoring)
print('10-fold cross validation average accuracy:%0.3f' %(results.mean()))'''

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(test_y, y_pred)
print(confusion_matrix)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

values1 = array(y_pred)
y_pred_label_encoder=LabelEncoder()
y_pred_Integer_encoded=y_pred_label_encoder.fit_transform(values1)
y_pred_onehot_encoder=OneHotEncoder(sparse=False)
y_pred_Integer_encoded=y_pred_Integer_encoded.reshape(len(y_pred_Integer_encoded),1)
y_pred_onehot_encoded=y_pred_onehot_encoder.fit_transform(y_pred_Integer_encoded)

values = array(test_y)
y_test_label_encoder=LabelEncoder()
y_test_Integer_encoded=y_test_label_encoder.fit_transform(values)
y_test_onehot_encoder=OneHotEncoder(sparse=False)
y_test_Integer_encoded=y_test_Integer_encoded.reshape(len(y_test_Integer_encoded),1)
y_test_onehot_encoded=y_test_onehot_encoder.fit_transform(y_test_Integer_encoded)

'''logit_roc_auc = roc_auc_score(y_test_onehot_encoded,y_pred_onehot_encoded)
fpr, tpr, thresholds = roc_curve(test_y, logreg.predict_proba(test_x)[:,1], pos_label='yes')
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' %logit_roc_auc)'''

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
clf_gini.fit(train_x, train_y)
y_pred_gini = clf_gini.predict(test_x)
print("Predictions using GINI index:")
print("Predicted Values:")
print(y_pred_gini)
print("Confusion Matrix: ")
print(confusion_matrix(test_y, y_pred_gini))
print("Accuracy: ")
print(accuracy_score(test_y, y_pred_gini)*100)
print("Detailed Report using GINI Index: ")
print(classification_report(test_y, y_pred_gini))
np.savetxt("D:/Kaggle-DataSet/Barcelona-Dataset/accidents_2017/2017_pred_GINI.csv",y_pred_gini,delimiter=',')
from sklearn import tree
tree.export_graphviz(clf_gini,out_file='D:/Kaggle-DataSet/Barcelona-Dataset/accidents_2017/Gini.dot')

clf_entropy=DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)
clf_entropy.fit(train_x,train_y)
y_pred_entropy=clf_entropy.predict(test_x)
print("Predictions using ENTROPY index:")
print("Predicted Values:")
print(y_pred_entropy)
print("Confusion Matrix: ")
print(confusion_matrix(test_y, y_pred_entropy))
print("Accuracy: ")
print(accuracy_score(test_y, y_pred_entropy)*100)
print("Detailed Report using ENTROPY Index: ")
print(classification_report(test_y, y_pred_entropy))
np.savetxt("D:/Kaggle-DataSet/Barcelona-Dataset/accidents_2017/2017_pred_ENTROPY.csv",y_pred_entropy,delimiter=',')
from sklearn import tree
tree.export_graphviz(clf_entropy,out_file='D:/Kaggle-DataSet/Barcelona-Dataset/accidents_2017/Entropy.dot')

from imblearn.over_sampling import SMOTE
cols = [i for i in df_acc.columns if i  not in Id_col+target_col]
cols=cols[5:]
smote_X=df_acc[cols]
smote_Y=df_acc[target_col]
smote_train_x,smote_test_x,smote_train_y,smote_test_y=train_test_split(smote_X,smote_Y,test_size=.20,
                                                                            random_state=100)
'''os=SMOTE(random_state=0)
os_smote_X,os_smote_Y=os.fit_sample(smote_train_x,smote_train_y)
os_smote_X=pd.DataFrame(data=os_smote_X,columns=cols)
os_smote_Y=pd.DataFrame(data=os_smote_Y,columns=target_col)'''

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
select=SelectKBest(score_func=chi2,k=3)
fit=select.fit(smote_X,smote_Y)

score=pd.DataFrame({"features":cols,"scores":fit.scores_,"p-values":fit.pvalues_})
score=score.sort_values(by="scores", ascending = False)
#Adding new columne "Feature Type in score dataframe
score["feature_type"]=np.where(score["features"].isin(num_cols),"Numerical","Categorical")
trace=go.Scatter(x=score[score["feature_type"]=="Categorical"]["features"],
                 y=score[score["feature_type"]=="Categorical"]["scores"],
                 name='Categorical', mode="lines+markers",
                 marker=dict(color='red',
                             line=dict(width=1))
                             )
trace1=go.Bar(x=score[score["feature_type"]=="Numerical"]["features"],
              y=score[score["feature_type"]=="Numerical"]["scores"],name='Numerical',
              marker=dict(color='royalblue',
                          line=dict(width=1)),
              xaxis='x2',yaxis='y2')

layout=go.Layout(dict(title="Scores of Importance for Categorical & Numerical features",
                      plot_bgcolor='rgb(243,243,243)',
                      paper_bgcolor='rgb(243,243,243)',
                      xaxis=dict(gridcolor='rgb(255,255,255)',
                                 tickfont=dict(size=10),
                                 domain=[0,0.7],
                                 tickangle=90, zerolinewidth=1,
                                 ticklen=5, gridwidth=2),
                      yaxis=dict(gridcolor='rgb(255,255,255)',
                                 title="scores",
                                 zerolinewidth=1, ticklen=5, gridwidth=2),
                      margin=dict(b=200),
                      xaxis2=dict(domain=[0.8,1], tickangle=90,
                                  gridcolor='rgb(255,255,255)'),
                      yaxis2=dict(anchor="x2",gridcolor='rgb(255,255,255)')))
data=[trace, trace1]
fig=go.Figure(data=data, layout=layout)
py.iplot(fig)

id_col=['Id']
target_col=['Victims']
cols=[i for i in df_acc.columns if i not in Id_col + target_col]
cols=cols[5:]
x=df_acc[cols]
X=np.array(x)
y=df_acc[target_col]
train_x,test_x,train_y,test_y = train_test_split(X,y,test_size=0.2,random_state=100)

#Random Forest Estimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
#import pydot
rf = RandomForestRegressor(n_estimators = 1000)
#Train the model on training data
rf.fit(train_x, train_y)
pred_rf = rf.predict(test_x)
#pred_rf = pd.DataFrame(pred_rf)
errors = abs(pred_rf - test_y)
print('Mean Absolute Error: ', round(np.mean(errors), 2), "degrees")
tree=rf.estimators_[100]
np.savetxt("D:/Kaggle-DataSet/Barcelona-Dataset/accidents_2017/2017_pred_RF.csv",pred_rf,delimiter=',')

from sklearn import tree
tree.export_graphviz(tree,out_file='D:/Kaggle-DataSet/Barcelona-Dataset/accidents_2017/Random_Forest.dot')


np.savetxt("D:/Kaggle-DataSet/Barcelona-Dataset/accidents_2017/2017_predicted.csv",test_y,delimiter=',')
#Predictive resutls for all Regressions:
#For Linear Regression - model reg and y_pred
#For KNN Regression - model KNN and y_pred_KNN
#For Decision Tree (GINI) - model clf_gini and y_pred_gini
#For Decision Tree (ENTROPY) - model clf_entropy and y_pred_entropy
#For Random Forest - model rf and pred_rf

#Model Report

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, scorer
from sklearn.metrics import f1_score
import statsmodels.api as sm
from sklearn.metrics import precision_score, recall_score
from yellowbrick.classifier import DiscriminationThreshold
import sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score

def model_report(model, training_x,testing_x,training_y,testing_y,name):
    model.fit(training_x,training_y)
    predictions=model.predict(testing_x)
    MSE=mean_squared_error(testing_y,predictions)
    R2=r2_score(testing_y,predictions)
    MAE=mean_absolute_error(testing_y,predictions)
    EVS=explained_variance_score(testing_y,predictions)
    
    df=pd.DataFrame({"Model"            :[name],
                     "Mean Sq.Error"   :[MSE],
                     "R-Square"     :[R2],
                     "Mean Abs.Error"        :[MAE],
                     "Variance Score"         :[EVS],
                     })
    return df
model1=model_report(reg,train_x,test_x,train_y,test_y,
                    "Linear Regression")
model2=model_report(KNN,train_x,test_x,train_y,test_y,"KNN Regression")
model3=model_report(clf_gini,train_x,test_x,train_y,test_y,"Decision Tree (GINI)")
model4=model_report(clf_entropy,train_x,test_x,train_y,test_y,"Decision Tree(Entropy)")
model5=model_report(rf,train_x,test_x,train_y,test_y,"Random Forest Regression")

model_performance=pd.concat([model1,model2,model3,model4,model5],axis=0).reset_index()
model_performance=model_performance.drop(columns="index",axis=1)
table=ff.create_table(np.round(model_performance,4))
py.iplot(table)

def output_tracer(metric, color):
    tracer=go.Bar(y=model_performance["Model"],
                  x=model_performance[metric],
                  orientation='h', name=metric,
                  marker=dict(line=dict(width=.7),
                              color=color))
    return tracer
layout=go.Layout(dict(title="Model Performances",
                      plot_bgcolor='rgb(243,243,243)',
                      paper_bgcolor='rgb(243,243,243)',
                      xaxis=dict(gridcolor='rgb(255,255,255)',
                                 title='metric',
                                 zerolinewidth=1,
                                 ticklen=5, gridwidth=2),
                      yaxis=dict(gridcolor='rgb(255,255,255)',
                                 zerolinewidth=1,
                                 ticklen=5, gridwidth=2),
                      margin=dict(l=250),
                      height=700))
                      
trace1=output_tracer("Mean Sq.Error",'#6699FF')
trace2=output_tracer('R-Square', 'red')
trace3=output_tracer('Mean Abs.Error','#33CC99')
trace4=output_tracer('Variance Score', 'lightgrey')

data=[trace1,trace2,trace3,trace4]
fig=go.Figure(data=data, layout=layout)
py.iplot(fig)

