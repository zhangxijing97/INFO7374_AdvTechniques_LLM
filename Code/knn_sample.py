import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Input Dateset
org_df = pd.read_csv("diabetes.csv")

#Define features and label for KNN
label_df =  org_df.loc[:,org_df.columns == 'Outcome']
feat_df =  org_df.loc[:,org_df.columns != 'Outcome']

##Encoding Categorical Variables
feat_df= pd.get_dummies(feat_df, dtype='int')

#Normalize Data
feat_df = (feat_df - feat_df.mean()) / feat_df.std()

#Seperate test and train data
train_x,test_x,train_y,test_y = train_test_split(feat_df,label_df,test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(train_x, train_y)
test_pred_y = knn.predict(test_x)

# accuracy measures
test_accuracy=accuracy_score(test_pred_y,test_y)
print(test_accuracy)