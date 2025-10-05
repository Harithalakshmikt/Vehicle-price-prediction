
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
x=car.drop(columns = 'Price')
y=car['Price']
