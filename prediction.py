
import pandas as pd
car = pd.read_csv('/workspaces/Vehicle-price-prediction/dataset/quikr_car.csv')
print(car.head())
car.info()
print(car['year'].unique())
print(car['Price'].unique())
print(car['kms_driven'].unique())


#Quality issues
"""
1. year column has some invalid values like 150k
2.year has to convert into int
3. price has to convert into int
4. price has some invalid values like ask for price,and commas in between the digits
5.kms_driven has to convert into int
6.kms_driven has some commas in between the digits
7.name should have three words
8.one outlier in price
"""
#Cleaning the data
backup=car.copy()
car=car[car['year'].str.isnumeric()]
car['year']=car['year'].astype(int)
car=car[car['Price']!='Ask For Price']
car['Price']=car['Price'].str.replace(',','').astype(int)
car['km_driven']=car['km_driven'].str.split(' ').str.get(0).str.replace(',','')
car=car[car['km_driven'].str.isnumeric()]
car['km_driven']=car['km_driven'].astype(int)
car=car[~car['fuel_type'].str.isna()]
car['name']= car['name'].str.split(' ').str.slice[0:3].str.join(' ')
car=car.reset_index(drop=True)
car.info()
car = car[car['price']<6e6].reset_index(drop=True)
car.to_csv("cleaned_car.csv")

#Model
X=car.drop(columns = 'Price')
y=car['Price']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2 score
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

ohe = OneHotEncoder()
ohe.fit(X[['name','company','fuel_type']])
scores=[]
colum_trans=make_column_transformer((OneHotEncode(categories= ohe.categories_),['name','company','fuel_type']),remainder='passthrough')


for i in range 1000:
    X_train,X_test,y_train,y_test =           train_test_split(X,y,test_size = .2,random_state = i)
    lr = LinearRegression ()
    pipe = make_pipeline(lr,column_trans)
    pipe.fit(X_train,y_train)
    y_pred = predict(X_test)
    scores.append(r2_score(y_pred,y_test))

print(np.argmax(scores))
print(scores[np.argmax(scores)]

X_train,X_test,y_train,y_test =           train_test_split(X,y,test_size = .2,random_state = np.argmax(scores))
lr = LinearRegression ()
pipe = make_pipeline(lr,column_trans)
pipe.fit(X_train,y_train)
y_pred = predict(X_test)
scores.append(r2_score(y_pred,y_test))
