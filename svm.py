# importing libraries  
import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd  
  
#importing datasets  
data_set= pd.read_csv('user_data.csv')  
  
#Extracting Independent and dependent Variable  
x= data_set.iloc[:, [2,3]].values  
y= data_set.iloc[:, 4].values  
  
# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)  
#feature Scaling  
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)   

    from sklearn.svm import SVC # "Support vector classifier"  
    classifier = SVC(kernel='linear', random_state=0)  
    classifier.fit(x_train, y_train)      
       #Predicting the test set result  
    y_pred= classifier.predict(x_test)  
    #Creating the Confusion matrix  
    from sklearn.metrics import confusion_matrix  
    cm= confusion_matrix(y_test, y_pred)  
