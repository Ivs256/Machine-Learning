import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()
df=pd.read_csv("sales_data.csv.csv")
y=df['GPA']
x1=df['SAT']
plt.scatter(x1,y)
yhat=0.0017*x1+0.275
fig=plt.plot(x1,yhat,lw=6,c='green',label='regression line')
plt.xlabel('SAT',fontsize=20)
plt.ylabel('GPA',fontsize=20)
plt.show()
x=sm.add_constant(x1)
results=sm.OLS(y,x).fit()
print(results.summary())
'''import polars as pl
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle
data = pl.read_csv('homeprices.csv')
X = data[['area']]
y = data['price']
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)
 #print(predictions)
# with open('model_pickle','wb') as f:
#     pickle.dump(model,f)
# with open('model_pickle','rb') as f:
#     mp=pickle.load(f)
plt.scatter(X, y, color='blue', label='Actual Prices')
plt.plot(X, predictions, color='red', label='Regression Line')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price')
plt.title('Linear Regression of Home Prices')
plt.legend()
plt.show()
# print(mp.predict([[5000]]))'''