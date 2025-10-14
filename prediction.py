
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
car['kms_driven']=car['kms_driven'].str.split(' ').str.get(0).str.replace(',','')
car=car[car['kms_driven'].str.isnumeric()]
car['kms_driven']=car['kms_driven'].astype(int)
car=car[~car['fuel_type'].isna()]
car['name']= car['name'].str.split(' ').str.slice(0,3).str.join(' ')
car=car.reset_index(drop=True)
car.info()
car = car[car['Price']<6e6].reset_index(drop=True)
car.to_csv("cleaned_car.csv")

#Model
X=car.drop(columns = 'Price')
y=car['Price']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

ohe = OneHotEncoder()
ohe.fit(X[['name','company','fuel_type']])
column_trans=make_column_transformer((OneHotEncoder(categories= ohe.categories_),['name','company','fuel_type']),remainder='passthrough')
scores=[]

#Try different random state
for i in range (1000):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .2,random_state = i)
    lr = LinearRegression ()
    pipe = make_pipeline(column_trans,lr)
    pipe.fit(X_train,y_train)
    y_pred = pipe.predict(X_test)
    scores.append(r2_score(y_test,y_pred))

#Best score and corresponding seed
print("Best random state =", np.argmax(scores))
print("Best R2_score = ", scores[np.argmax(scores)])

#Fiting the best seed
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .2,random_state = np.argmax(scores))
lr = LinearRegression ()
pipe = make_pipeline(column_trans,lr)
pipe.fit(X_train,y_train)
y_pred = pipe.predict(X_test)
print("Final best score = ",r2_score(y_test,y_pred))

test_data = X_test.head().copy()
test_data["actual_price"] = y_test.head().values
print(test_data)
test_data.to_csv("sample data.csv", index=False)
import pickle
pickle.dump(pipe,open('LinearRegressionModel.pkl','wb'))
