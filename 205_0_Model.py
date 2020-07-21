import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import pyplot
from datetime import *
from sklearn.model_selection import train_test_split
from datetime import datetime
from datetime import date
import random
import seaborn as sns
from sklearn.metrics import accuracy_score,mean_squared_error,r2_score,mean_absolute_error
import lightgbm as lgb
import xgboost as xgb

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

def accuracy_rate(list_real,list_predict,boundary):
    numerator=0
    for i,j in zip(list_real,list_predict):
        if abs(i-j) <=boundary:
            numerator+=1
    denominator = len(list_real)
    return str(100*(1.0*numerator/denominator))+'%'

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def weighted_absolute_percentage_error(y_true, y_pred): 
    #y_true=[ sum(y_true)/len(y_true)  for i in range(len(y_true))] 
    #print ('平均值 : ',y_true[0])
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    #return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return np.sum(np.abs((y_true - y_pred)))/np.sum(y_true) * 100

##load data
train = pd.read_csv('train_205_0.csv', engine='python', encoding='big5', dtype = str)
tripid_data = pd.read_excel('發車時刻表.xlsx', dtype = str)


##change data type_train
train['H_1R'] = train['H_1R'].astype('float64')
train['TEMP'] = train['TEMP'].astype('float64')
train['Byyyymmdd'] = train['Byyyymmdd'].astype('int64')
train['StopSequence'] = train['StopSequence'].astype('int64')
train['month'] = train['month'].astype('int64')
train['Direction'] = train['Direction'].astype('int64')
train['big_holiday'] = train['big_holiday'].astype('float64')
train['PASSNUM'] = train['PASSNUM'].astype('int64')

##change data type_train_tripid_data
tripid_data['TripID'] = tripid_data['TripID'].astype('int64')
tripid_data['Direction'] = tripid_data['Direction'].astype('int64')

##Data preprocessing
train['降雨量'] = train['H_1R']
train['是否下雨'] = np.where(train['降雨量']>0, 1, 0)
train['typhoon'] = train['typhoon'].map({'Y':'1', 'N':'0'})
train['first_day'] = train['first_day'].map({'Y': 1, 'N': 0})
train['weekend'] = train['weekend'].map({'Y':'1', 'N':'0'})
train['Date'] = pd.to_datetime(train['Byyyymmdd'], format='%Y%m%d')
train['WEEKDAY'] = train['Date'].dt.strftime('%A')
pd.get_dummies(train.WEEKDAY) 
train = train.join(pd.get_dummies(train.WEEKDAY))
train.loc[train.TEMP <0, 'TEMP'] = 26
train['TEMP'].fillna(value=0, inplace=True)
train['HOLIDAY'] = train['weekend'].map({'1': 2, '0': 1})
train['HOLIDAY'] = np.where(train.big_holiday == 1, 0, train.HOLIDAY)
train.Byyyymmdd.unique()
train['Break'] = np.where((train['Byyyymmdd'] > 20190120) & (train['Byyyymmdd'] < 20190211)|(train['Byyyymmdd'] > 20190630) & (train['Byyyymmdd'] < 20190830)|(train['Byyyymmdd'] > 20200120) & (train['Byyyymmdd'] < 20200226), 1, 0)

##Data merge
Train = pd.merge(train, tripid_data, how='left', on=['Bmmdd', 'RouteName_Zh_tw', 'Direction'])
Train['StopSequence_START'] = Train['StopSequence']
Train['StopName_START'] = Train['StopName_Zh_tw']

##Remove outliers
from scipy import stats
print(len(Train))
train1 = pd.DataFrame().copy()
for q in Train['Break'].unique():
    for k in Train['HOLIDAY'].unique():
        for j in Train['TripID'].unique():
            for i in Train['StopSequence'].unique():
                print(q, k, j, i)
                train = Train[(Train['Break']== q)]
                train = train[(train['HOLIDAY']== k)]
                train = train[(train['TripID']== j)]
                train = train[(train['StopSequence']== i)]
                numeric_train=train[['PASSNUM']]
                z=np.abs(stats.zscore(numeric_train))
                threshold=2
                train_nooutlier=train[(z<2).all(axis=1)]
                train1 = train1.append(train_nooutlier)
            
Train = train1.copy()
print(Train.groupby('StopSequence')['StopSequence'].count())
print(len(Train))
Train.to_csv('Train_no_205.csv', index = True, encoding='ansi')

##Reload data
Train = pd.read_csv('Train_no_205.csv', engine='python', encoding='big5')

#Add rain_level variable
Train['rain_level'] = np.where(Train['降雨量']>40, 2, 1)
Train['rain_level'] = np.where(Train['降雨量']<10, 0, Train['rain_level'])

##Data merge_2
Geoinfo = pd.read_excel('Geoinfo3.xlsx')
Geoinfo['StopSequence_START'] = Geoinfo['StopSequen']
Train = pd.merge(Train, Geoinfo, how='left', on=['SubRouteID', 'Direction', 'StopSequence_START'])
Train['Tran_score'] = Train['MRT']*2 + Train['TRA']*7 + Train['LRT']*1 + Train['BUS']*4
Train['ACT_score'] = Train['Makert']*2 + Train['Hospital']

#Split the data
train1 = pd.DataFrame().copy()
train1 = train1.append( Train.sample(frac=0.8, replace=True, random_state=1))
    
dftrain = train1.copy()
dftest = Train.drop(dftrain.index).reset_index(drop = True)
dftrain = dftrain.reset_index(drop = True)
print('df={}, train={}, test={}'.format( len(Train), len(dftrain), len(dftest)) )


X_train = dftrain[['RouteName_Zh_tw', 'StopName_START', 'StopSequence_START', 'month', 'Direction', 'TripID', 'big_holiday', 'typhoon', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', '是否下雨', 'rain_level', 'Break', 'P_CNT', 'school_活動集結點', 'ACT_score', 'Tran_score', 'PASSNUM']]
X_test = dftest[['RouteName_Zh_tw', 'StopName_START', 'StopSequence_START', 'month', 'Direction', 'TripID', 'big_holiday', 'typhoon', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', '是否下雨', 'rain_level', 'Break', 'P_CNT', 'school_活動集結點', 'ACT_score', 'Tran_score', 'PASSNUM']]
print(X_train.columns[2:22])
print(X_test.columns[2:22])
train_col = X_train.columns[2:22]

Xtrain = X_train[train_col].values
Xtest = X_test[train_col].values
Ytrain = X_train['PASSNUM']
Ytest = X_test['PASSNUM']

##Model Training
model_2 = xgb.XGBRegressor(learning_rate = 0.4, n_estimators = 200, max_depth = 6, gamma = 0, min_child_weight = 1,
                colsample_bytree = 1, colsample_bylevel = 1, subsample = 1, reg_lambda = 1, reg_alpha = 0)
model_2.fit(np.array(Xtrain), np.array(Ytrain))

##Prediction 
y_predicted = model_2.predict(np.array(Xtest))
y_predicted= [ round(i)  for i in  y_predicted.tolist() ] 


##Evaluation
mse = mean_squared_error(Ytest, y_predicted ) #均方誤差
mae = mean_absolute_error(Ytest, y_predicted ) #平均絕對值誤差
mape = mean_absolute_percentage_error(Ytest, y_predicted)
wape = weighted_absolute_percentage_error(Ytest, y_predicted)
print ('RMSE = %.3f, MAE = %.3f' %(mse**0.5,mae))
Ytest_mean = sum(Ytest)/len(Ytest)
print ('MAPE = %.3f, WAPE = %.3f, 真實資料平均搭乘人數 = %.3f, 誤差人數 = %.3f' %(mape,wape,Ytest_mean,Ytest_mean*wape/100) )
print ('在容許上下人數誤差小於3個人的情況下，其準確度 = %s' %(accuracy_rate(Ytest.tolist(),y_predicted, 4)))
passnum_predict = y_predicted

MAPE = (np.abs((Ytest -  passnum_predict) / Ytest)) * 100
gap = np.abs(passnum_predict - Ytest)
d = {'PASSNUM':Ytest ,'predict_passnum':passnum_predict}
Testing = pd.DataFrame(d)
Testing['MAPE'] = (np.abs((Testing['PASSNUM'] -  Testing['predict_passnum']) / Testing['PASSNUM'])) * 100
Testing['gap'] = np.abs(Testing['predict_passnum'] - Testing['PASSNUM'])
Testing[['PASSNUM','predict_passnum', 'gap', 'MAPE']]
Testing.loc[Testing.gap <4, 'MAPE'] = 0
Testing[['PASSNUM','predict_passnum', 'gap', 'MAPE']]
print(Testing[~(Testing['MAPE']==np.inf)]['MAPE'].mean())

###features importances
df = pd.DataFrame(data = model_2.feature_importances_, index=['StopSequence_START', 'month', 'Direction', 'TripID', 'big_holiday', 'typhoon', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', '是否下雨', 'rain_level', 'Break', 'P_CNT', 'school_活動集結點', 'ACT_score', 'Tran_score'], columns=['importance'])
feature_importances = df.sort_values('importance',ascending=False)
feature_importances

##Multicollinearity
Train.head(5)
df = Train[['month', 'TripID', 'P_CNT', 'school_活動集結點', 'ACT_score', 'Tran_score', 'PASSNUM']]
df.head(5)
df.corr()
plt.figure(figsize = (20, 20))
ax=sns.heatmap(df.corr(),annot=True,fmt='.1g',cmap='coolwarm')
ax.get_ylim()

##Save the model
import pickle
pickle.dump(model_2, open("C:/Users/user/Desktop/X139/205/model_205_0.pickle.dat", "wb"))

##Testing
##Load the data
test = pd.read_csv('model_input_205_0_v2.csv',engine='python', encoding='big5', dtype = str)
test['Bmmdd'] = test['StartTime']

test['是否下雨'] = test['是否下雨'].astype('int64')
test['TEMP'] = test['TEMP'].astype('float64')

test['StopSequence_Start'] = test['StopSequence_START']
test['StopSequence_START_HIS'] = test['StopSequence_START_HIS'].replace(np.nan, 0)
test['StopSequence_START_HIS'] = test['StopSequence_START_HIS'].astype('int64')
test['StopSequence_START'] = test['StopSequence_START_HIS']
test['month'] = test['month'].astype('int64')
test['Direction'] = test['Direction'].astype('int64')
test['big_holiday'] = test['big_holiday'].astype('float64')
test['Sunday'] = test['Sunday'].astype('int64')
test['Monday'] = test['Monday'].astype('int64')
test['Tuesday'] = test['Tuesday'].astype('int64')
test['Wednesday'] = test['Wednesday'].astype('int64')
test['Thursday'] = test['Thursday'].astype('int64')
test['Friday'] = test['Friday'].astype('int64')
test['Saturday'] = test['Saturday'].astype('int64')
test['StopSequence_START'] = test['StopSequence_START'].astype('int64')
test['SubRouteID'] = test['SubRouteID'].astype('int64')
test['rain_level'] = np.where(test['降雨量']>40, 2, 1)
test['rain_level'] = np.where(test['降雨量']<10, 0, test['rain_level'])
test['Break'] = 0
##tripid
tripid_data = pd.read_excel('發車時刻表.xlsx', dtype = str)
tripid_data['TripID'] = tripid_data['TripID'].astype('int64')
tripid_data['Direction'] = tripid_data['Direction'].astype('int64')

Test = pd.merge(test, tripid_data, how='left', on=['Bmmdd', 'RouteName_Zh_tw', 'Direction'])

##Add_Geoinfo
Geoinfo = pd.read_excel('Geoinfo.xlsx')
Geoinfo['StopSequence_START'] = Geoinfo['StopSequen']

Test = pd.merge(Test, Geoinfo, how='left', on=['SubRouteID', 'Direction', 'StopSequence_START'])
Test['Tran_score'] = Test['MRT']*2 + Test['TRA']*7 + Test['LRT']*1 + Test['BUS']*4
Test['ACT_score'] = Test['Makert']*2 + Test['Hospital']


X_test = Test[['RouteName_Zh_tw', 'StopName_START', 'StopSequence_START', 'month', 'Direction', 'TripID', 'big_holiday', 'typhoon', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', '是否下雨', 'rain_level', 'Break', 'P_CNT', 'school_活動集結點', 'ACT_score', 'Tran_score']]
print(X_test.columns[2:])
test_col = X_test.columns[2:]
Xtest = X_test[test_col].values

y_predicted = model_2.predict(np.array(Xtest))
y_predicted= [ round(i)  for i in  y_predicted.tolist() ]
passnum_predict = y_predicted
Test['PASSNUM'] = np.abs(y_predicted)

##Add value to new stop
for i in Test['PASSNUM'][(Test['Is_New_Stop']=='Y')].index:
    a = Test['PASSNUM'][i-1].tolist()
    b = Test['PASSNUM'][i+1].tolist()
    data = np.array([a, b])
    c = np.average(data, axis=0).astype('int32')
    c1 = Test.loc[i,'PASSNUM'] = c
    print(i, a, b, c, c1)

df = pd.DataFrame(Test, columns= ['EnterpriseID', 'DATADATE', 'RouteName_Zh_tw', 'SubRouteID', 'Direction', 'StartTime', 'StopName_START', 'StopName_END', 'StopSequence_Start', 'StopSequence_END', 'PASSNUM', 'Created_By', 'Created_Date', 'Last_Updated_By', 'Ladt_Updated_Date'])

##Export the prediction result to csv file
Weekday_name = date.today().strftime("%A")
df[(Test[Weekday_name]== 1)].to_csv('205_0_PRED.csv', index = False, encoding='ansi')

