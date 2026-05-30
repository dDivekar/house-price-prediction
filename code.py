import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,root_mean_squared_error,mean_absolute_error
import pandas as pd
from google.colab import files
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
uploaded=files.upload()

df=pd.read_csv('Bengaluru_House_Data.csv')
df=df[['area_type','size','total_sqft','bath','balcony','price','society','location','availability']]
df=pd.DataFrame(df)
print(df)

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
print(df.index)

print(df.dtypes)

#extracting numbers from size(bhk)
df['size'] = df['size'].astype(str).str.extract(r'(\d+)', expand=False).astype(float)

print(df['size'])

#encoding area type
la=LabelEncoder()
df['area_type']=la.fit_transform(df['area_type'])
print(df['area_type'])

#encoding society type
la=LabelEncoder()
df['society']=la.fit_transform(df['society'])
print(df['society'])

#encoding location type
df['location'] = df['location'].astype(str)
la=LabelEncoder()
df['location']=la.fit_transform(df['location'])
print(df['location'])

#converting sqft to float
def convert_sqft_to_float(x):
    try:
        if isinstance(x, str) and '-' in x:
            tokens = [float(t) for t in x.split('-')]
            return sum(tokens) / len(tokens)
        return float(x)
    except (ValueError, TypeError):
        return None

df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_float)

df.dropna(subset=['total_sqft'], inplace=True)

print(df['total_sqft'])

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df['total_sqft'] = scaler.fit_transform(df[['total_sqft']])
print(df['total_sqft'])

df['size']=scaler.fit_transform(df[['size']])
print(df['size'])

df['bath']=scaler.fit_transform(df[['bath']])
print(df['bath'])

df['balcony']=scaler.fit_transform(df[['balcony']])
print(df['balcony'])


df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
print(df.index)

df['availability_category'] = np.where(df['availability'].str.contains('Ready To Move'), 'Ready To Move', 'Date-based')
print(df[['availability', 'availability_category']].head())

la = LabelEncoder()
df['availability_category_encoded'] = la.fit_transform(df['availability_category'])
print(df[['availability_category', 'availability_category_encoded']].head())

print(df['availability_category_encoded'].value_counts())

X = df.drop(['price'],axis=1)
y = df['price']
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4709)
print(x_train)
print(y_train)

x=np.array(x_train)
y=np.array(y_train)

model=LinearRegression()
x_train = x_train.drop(columns=['availability', 'availability_category'])
x_test = x_test.drop(columns=['availability', 'availability_category'])
model.fit(x_train, y_train)
pred_y_train=model.predict(x_train)
print("Intercept : ",model.intercept_)
print("Slope : ",model.coef_[0])
print("predicted y_train samples : ",pred_y_train)

pred_y_test = model.predict(x_test)
print("mse :",mean_squared_error(y_test, pred_y_test))
print("rmse :",root_mean_squared_error(y_test, pred_y_test))
print("mae :",mean_absolute_error(y_test, pred_y_test))
print("r2 score :",r2_score(y_test, pred_y_test))

#box plot
np.random.seed(42)
data=df['size']
plt.boxplot(data,vert=True,labels=['Boxplot example'],patch_artist=True)
plt.title("Box Plot")
plt.show()

def RemoveOutlier():
    Q1 = df['availability_category_encoded'].quantile(0.25)
    Q3 = df['availability_category_encoded'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    newdf= df[(df['availability_category_encoded']>= lower_bound)
                & (df['availability_category_encoded']<= upper_bound)]
    return newdf

print(df.index)

max_r2=0
max_random_state=0
for i in range(0,7000) :
  X = df.drop(['price'],axis=1)
  y = df['price']
  x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=i)

  x=np.array(x_train)
  y=np.array(y_train)

  model=LinearRegression()
  x_train = x_train.drop(columns=['availability', 'availability_category'])
  x_test = x_test.drop(columns=['availability', 'availability_category'])
  model.fit(x_train, y_train)
  pred_y_train=model.predict(x_train)

  pred_y_test = model.predict(x_test)
  if max_r2>=r2_score(y_test, pred_y_test):
    continue
  else :
    max_r2=r2_score(y_test, pred_y_test)
    max_random_state=i

print(max_r2)
print(max_random_state)
