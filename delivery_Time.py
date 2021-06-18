# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 15:53:02 2021

@author: Soumya PC
"""

#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading the dataset
delivery_time= pd.read_csv('delivery_time.csv')
delivery_time.columns
delivery_time.head()

#Visualization
plt.hist(delivery_time.SortingTime)
plt.hist(delivery_time.DeliveryTime)
plt.boxplot(delivery_time.SortingTime)
plt.boxplot(delivery_time.DeliveryTime) 
plt.plot(delivery_time.SortingTime,delivery_time.DeliveryTime,"bo");plt.xlabel('SortingTime');plt.ylabel('DeliveryTime')

#Correlation coeff
delivery_time.DeliveryTime.corr(delivery_time.SortingTime)
np.corrcoef(delivery_time.DeliveryTime,delivery_time.SortingTime)
#0.825

#Model building
import statsmodels.formula.api as smf
model=smf.ols('DeliveryTime~SortingTime',data=delivery_time).fit()
model.params
model.summary()
#Here R-Sqaured=0.68, Adj. R squared=0.66,P-value=0.001(Intercept),0.00(SortingTime)
#Conf_int
model.conf_int(0.05)
pred= model.predict(delivery_time)
pred

Error= Actual-Predicted
Error= delivery_time.DeliveryTime-pred

#RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(delivery_time.DeliveryTime,pred))
#Here rmse=2.79


#As the R sqaured value is less, we will transform the features for accuracy
#Transformation-log
model1=smf.ols('DeliveryTime~np.log(SortingTime)',data=delivery_time).fit()
model1.params
model1.summary()
#Here Rsquared=0.69,Aj.R squared=0.679,P=0.6(Intercept),0.00(SortingTime)
#Conf_int
model1.conf_int(0.05)
pred1=model1.predict(delivery_time)
pred1
#error
Error1=delivery_time.DeliveryTime-pred1
#RMSE
rmse1= sqrt(mean_squared_error(delivery_time.DeliveryTime,pred1))
#rmse=2.73,which is comparitively less than previos built model
#Rsquared and adj. R squaared is also high compared to the previous model built

#visualization(actual and pred values)
plt.scatter(x=delivery_time['SortingTime'],y=delivery_time['DeliveryTime'],color='green');plt.plot(delivery_time['SortingTime'],pred1,color='black');plt.xlabel('SortingTime');plt.ylabel('DELIVERYTIME')

#Exponential transformation
model2=smf.ols('np.log(DeliveryTime)~SortingTime',data=delivery_time).fit()
model2.params
model2.summary()
#Here, R-Squared= 0.71, Adj R squared=0.69, P is 0.00
model2.conf_int(0.05)
pred2=model2.predict(delivery_time)
pred2
Error2=delivery_time.DeliveryTime-pred2
#RMSE
rmse2= sqrt(mean_squared_error(delivery_time.DeliveryTime,pred2))
rmse2
#Here rmse=14.79

#Quadratic transformation
delivery_time['SortingTime_sq']=delivery_time.SortingTime*delivery_time.SortingTime
model3=smf.ols('DeliveryTime~SortingTime+SortingTime_sq',data=delivery_time).fit()
model3.params
model3.summary()
#Here R squared=0.69 Adj R squared=0.659 P is 0.42 for SortingTime_sq(which is greater than 0.05), so we will not consider this model3
pred3= model3.predict(delivery_time)
pred3

Error3=delivery_time.DeliveryTime-pred3
rmse3= sqrt(mean_squared_error(delivery_time.DeliveryTime,pred3))
rmse3
Here rmse=2.74

#Conclusion- We will consider the Log i.e model1 to be better model, as the R-squared=0.69, Adj RSquared= 0.679 and P value is 0.00 which is less than 0.005 and RMSE= 2.73 which is lesser compared to all the transformed models.