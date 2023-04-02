# -*- coding: utf-8 -*-


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('SalaryData\SalaryData.csv') 

features = df.drop(['Salary','Employee ID'],axis = 1)

features=[i for i in (features.iloc[:,:]).columns if features[i].dtypes!='object' ]


i = 1
plt.figure(figsize=(28,5))

for j in features:
    x=df[j].values
    y=df['Salary'].values
    plt.subplot(1, len(features), i)
    plt.scatter(x,y,label=j, color='r')
    plt.xlabel(j)
    plt.ylabel('Salary')
    i += 1
    

#Correlation Matrix to get the relavent features
try:
    plt.figure(figsize=(15,5))
    plt.matshow(sns.heatmap(df.corr()))
    plt.show()
except :
    pass


#Drop irrelevant features and features with correlation less than +-0.15 with the salary

X = df.drop(['Employee ID','Salary','Education','Project Involvement','Over 18 years old','Job Level'],axis = 1)
y = df['Salary']


# encode categorical variables
cat_var=[var for var in X.columns if X[var].dtype=='O']


num_var=[var for var in X.columns if var not in cat_var]

enc = OneHotEncoder()

enc_data=pd.DataFrame(enc.fit_transform(X[cat_var]).toarray())

X=X.join(enc_data)

X = X.drop(cat_var,axis = 1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# creating a regression model
model = LinearRegression()

# fitting the model
model.fit(X_train,y_train)

predictions = model.predict(X_test)


# model evaluation
print(
  'mean_squared_error : ', mean_squared_error(y_test, predictions))
print(
  'mean_absolute_error : ', mean_absolute_error(y_test, predictions))


employee_input = {'Age':36 ,"Gender":'Male','Job Role':'Business Developer','Total Working Years': 8,'Years At Company':8 ,'Years In Current Role':8}

employee_input= pd.DataFrame(employee_input,index=[0])

enc_input=pd.DataFrame(enc.transform(employee_input[cat_var]).toarray())

employee_input=employee_input.join(enc_input)

employee_input = employee_input.drop(cat_var,axis = 1)



employee_prediction = model.predict(employee_input)
print("Employee predicted Salary : ",employee_prediction)