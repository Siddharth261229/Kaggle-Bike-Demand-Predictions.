import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# 1 Read the data

bikes = pd.read_csv("C:/Users/hp/Downloads/hour.csv")

# Prelim analysis and feature seletions

bikes_prep = bikes.copy()
bikes_prep = bikes_prep.drop(['index','date','casual','registered'], axis=1)

#2  Basic checks and missing values

bikes_prep.isnull().sum()

# visualise the data using panda histogrm

bikes_prep.hist(rwidth=0.9)
plt.tight_layout()

# 3 Data visualisation
# visualise the continuous features vs demand

plt.subplot(2,2,1)
plt.title("temp vs demand")
plt.scatter(bikes_prep['temp'],bikes_prep['demand'] 
            , s=2 , c='g')


plt.subplot(2,2,2)
plt.title("atemp vs demand")
plt.scatter(bikes_prep['atemp'],bikes_prep['demand'] 
            , s=2, c='b')


plt.subplot(2,2,3)
plt.title("humidity vs demand")
plt.scatter(bikes_prep['humidity'],bikes_prep['demand'] 
            , s=2,c='m')


plt.subplot(2,2,4)
plt.title("windspeed vs demand")
plt.scatter(bikes_prep['windspeed'],bikes_prep['demand'] 
            , s=2, c='c')


# plot the categorical features vs demand
# create a 3*3 subplot

plt.subplot(3,3,1)
plt.title('average demand per season')
# create alist of unique season's values
cat_list = bikes_prep['season'].unique()


# create avergae demand per season using groupby
cat_average = bikes_prep.groupby('season').mean()['demand']
colours = ['g','r','m','b']
plt.bar(cat_list, cat_average, color =colours)

plt.subplot(3,3,2)
plt.title('average demand per month')
cat_list= bikes_prep['month'].unique()
cat_average= bikes_prep.groupby('month').mean()['demand']
plt.bar(cat_list, cat_average, color=colours)


plt.subplot(3,3,3)
plt.title('average demand per weekday')
cat_list= bikes_prep['weekday'].unique()
cat_average= bikes_prep.groupby('weekday').mean()['demand']
plt.bar(cat_list, cat_average, color=colours)


plt.subplot(3,3,4)
plt.title('average demand per holiday')
cat_list= bikes_prep['holiday'].unique()
cat_average= bikes_prep.groupby('holiday').mean()['demand']
plt.bar(cat_list, cat_average, color=colours)


plt.subplot(3,3,5)
plt.title('average demand per year')
cat_list= bikes_prep['year'].unique()
cat_average= bikes_prep.groupby('year').mean()['demand']
plt.bar(cat_list, cat_average, color=colours)


plt.subplot(3,3,6)
plt.title('average demand per hour')
cat_list= bikes_prep['hour'].unique()
cat_average= bikes_prep.groupby('hour').mean()['demand']
plt.bar(cat_list, cat_average, color=colours)


plt.subplot(3,3,7)
plt.title('average demand per workingday')
cat_list= bikes_prep['workingday'].unique()
cat_average= bikes_prep.groupby('workingday').mean()['demand']
plt.bar(cat_list, cat_average, color=colours)


plt.subplot(3,3,8)
plt.title('average demand per weather')
cat_list= bikes_prep['weather'].unique()
cat_average= bikes_prep.groupby('weather').mean()['demand']
plt.bar(cat_list, cat_average, color=colours)

plt.tight_layout()


# check for outliers

bikes_prep['demand'].describe()

bikes_prep['demand'].quantile([0.05,0.1,0.15,0.9,.95,.99])

#4 check for mulitple linear regression assumptions

#linearity using correlation coefficient matrix using corr

correlation = bikes_prep[['temp','atemp','humidity','windspeed','demand']].corr()

bikes_prep = bikes_prep.drop(['weekday','year','workingday','atemp','windspeed'],axis=1)

# check the autocorrelation in demand using acorr

df1= pd.to_numeric(bikes_prep['demand'], downcast='float')
plt.acorr(df1, maxlags=12)

# create/modify new features

# log normalise the feature 'demand'

df1 = bikes_prep['demand']
df2 = np.log(df1)

plt.figure()
df1.hist(rwidth= 0.9 , bins = 20 )

plt.figure()
df2.hist(rwidth = 0.9 , bins=20)

bikes_prep['demand'] = np.log(bikes_prep['demand'])

# autocorrelation in the demand column

t_1 = bikes_prep['demand'].shift(+1).to_frame()
t_1.columns = ['t-1']

t_2 = bikes_prep['demand'].shift(+2).to_frame()
t_2.columns = ['t-2']

t_3 = bikes_prep['demand'].shift(+3).to_frame()
t_3.columns = ['t-3']

bikes_prep_lag = pd.concat([bikes_prep,t_1,t_2,t_3], axis=1)

bikes_prep_lag = bikes_prep_lag.dropna()

# create dummy variables and drop first
# to avoid dummy varibles trap using get_dummies
# - season , holiday, weather, month, hour

bikes_prep_lag['season'] = bikes_prep_lag['season'].astype('category')
bikes_prep_lag['holiday'] = bikes_prep_lag['holiday'].astype('category')
bikes_prep_lag['weather'] = bikes_prep_lag['weather'].astype('category')
bikes_prep_lag['month'] = bikes_prep_lag['month'].astype('category')
bikes_prep_lag['hour'] = bikes_prep_lag['hour'].astype('category')

bikes_prep_lag = pd.get_dummies(bikes_prep_lag, drop_first=True)

# create train and test split



#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = \
 #   train_test_split(x,y, test_size= 0.4,  random_state=1234)


#demand is time  dependent

Y = bikes_prep_lag[['demand']]
X = bikes_prep_lag.drop(['demand'], axis=1)

tr_size = 0.7 * len(X)
tr_size = int(tr_size)

X_train = X.values[0 : tr_size]
X_test = X.values[tr_size : len(X)]

Y_train = Y.values[0 : tr_size]
Y_test = Y.values[tr_size : len(Y)]

# prediction using multiple linear regression
from sklearn.linear_model import LinearRegression

std_reg = LinearRegression()
std_reg.fit(X_train, Y_train)

r2_train = std_reg.score(X_train, Y_train)
r2_test  = std_reg.score(X_test, Y_test)


# Create Y Predictions
Y_predict = std_reg.predict(X_test)


from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(Y_test, Y_predict))


# Calculate RMSLE and compare results
# Exponent opposite to log

Y_test_e = []
Y_predict_e = []

for i in range(0, len(Y_test)):
    Y_test_e.append(math.exp(Y_test[i]))
    Y_predict_e.append(math.exp(Y_predict[i]))

log_sq_sum = 0.0

for i in range(0, len(Y_test_e)):
    log_a = math.log(Y_test_e[i] + 1)
    log_p = math.log(Y_predict_e[i] + 1)
    log_diff = (log_p - log_a)**2
    log_sq_sum = log_sq_sum + log_diff

rmsle = math.sqrt(log_sq_sum/len(Y_test))

print(rmsle)
