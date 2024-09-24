import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression

#Input Dateset
org_df = pd.read_csv("house_ds.csv")

#Define features and outcome for Regression
outcome_df =  org_df.loc[:,org_df.columns == 'price']
feat_df =  org_df.loc[:,org_df.columns == 'sqft_living']

# #Log Transform
# outcome_df['price'] = np.log(outcome_df['price'])
# feat_df['sqft_living'] = np.log(feat_df['sqft_living'])

#Seperate test and train data
train_x,test_x,train_y,test_y = train_test_split(feat_df,outcome_df,test_size=0.25)

#Create a Reg model
model = LinearRegression()
model.fit(train_x,  train_y)
print('slope=',model.coef_)
print('intercept=',model.intercept_)

#test_pred_y = model.coef_ * test_x + model.intercept_
test_pred_y = model.predict(test_x)

#Visualize the model
plt.scatter(test_x, test_y)
plt.plot(test_x, test_pred_y)
plt.show()

# accuracy measures
r_sq = model.score(test_x, test_y)
print(r_sq)



#Input Datesets
org_df = pd.read_csv("diabetes.csv")

#Define features and outcome for Regression
outcome_df =  org_df.loc[:,org_df.columns == 'Outcome']
feat_df =  org_df.loc[:,org_df.columns != 'Outcome']

#Seperate test and train data
train_x,test_x,train_y,test_y = train_test_split(feat_df,outcome_df,test_size=0.25)

#Create a Reg model
model = LogisticRegression()
model.fit(train_x,  train_y)

#accuracy measures
test_accuracy=model.score(test_x,test_y)
print(test_accuracy)