
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
"""
#Cleaning the data
backup=car.copy()
car=car[car['year'].str.isnumeric()]
car['year']=car['year'].astype(int)
car=car[car['Price']!='Ask For Price']
