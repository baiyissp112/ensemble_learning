# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 10:00:28 2021

@author: Administrator
"""
import time
import datetime
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.model_selection import GridSearchCV







#-----------------------读取csv，划分训练测试集----------------------------
pd_data = pd.read_csv(r'K:\csv\2019_train_delete_smogn_gas.csv',encoding='gb2312')

tic = time.time()

X = pd_data.loc[:, ('month','day','lon','lat',
                    'CO','HCHO','NO2',
                    'vieo','vino',
                    'sp','rh','blh','ssrd','t2m','u10','v10','cbh','alnid')]
#X = pd_data.loc[:, ('month','day','lat','lon','blh','r','sp','ssrd','t2m','tcc','tp','u10','v10','CO','HCHO','NO2','SO2')]

Y = pd_data.loc[:, 'O3']
#y = pd_data.loc[:, 'pm25']


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

#y_train = y_train.astype('int')



'''
#-----------------------调整模型参数----------------------------
ScoreAll = []
for i in range(1,10,1):
    clf_e = ExtraTreesRegressor(n_estimators=180,max_depth=22,min_samples_split=3,min_samples_leaf=1)
    clf_e.fit(X_train, y_train)
    
    model_result = pd.Series(clf_e.predict(X_test))
    score = r2_score(y_test,model_result)
    
    print('min_samples_leaf :',i,'R2: %.4f'%score)
    ScoreAll.append([i,score])
   
ScoreAll = np.array(ScoreAll)
max_score = np.where(ScoreAll==np.max(ScoreAll[:,1]))[0][0] ##这句话看似很长的，其实就是找出最高得分对应的索引
print("最优参数以及最高得分:",ScoreAll[max_score])  
plt.figure(figsize=[20,5])
plt.plot(ScoreAll[:,0],ScoreAll[:,1])
plt.show()
'''



#-----------------------选择到合适的模型参数，开始训练----------------------------
clf_e = ExtraTreesRegressor()#n_estimators=300,max_depth =30,min_samples_split=6,min_samples_leaf= 1
clf_e.fit(X_train, y_train)



#保存模型
#with open(r'K:\model\ERF\ERF_2019_smogn.pickle','wb') as fw:
    #pickle.dump(clf_e,fw)

#预测
y_predict = pd.Series(clf_e.predict(X_test))

#本来是乱序重新索引
y_test=y_test.reset_index(drop=True)




#计算指标
r2 = r2_score(y_test,y_predict)
rmse = np.sqrt(mean_squared_error(y_test,y_predict))
mae = mean_absolute_error(y_test,y_predict)




'''
x = range(len(y_predict))
plt.plot(y_predict,color='b',label='model_value')
plt.plot(y_test,color='r',label='station_value')
plt.xlim(0,200)
plt.legend()
plt.show()

'''


from scipy.stats import gaussian_kde
# Calculate the point density
xy = np.vstack([y_test,y_predict])
z = gaussian_kde(xy)(xy)



plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False

fig,ax  = plt.subplots()

density = plt.scatter(y_test, y_predict, c=z, s=0.3,vmin=0,vmax=0.0003)#


plt.colorbar(density,label='高斯概率密度')


a,b = np.polyfit(y_test,y_predict,1)
plt.plot([0,400],[b,a*400+b],color='blue',linestyle='--')#拟合线
plt.plot([0,400],[0,400],color='red',linestyle='--')# y=x线

#plt.title('r2:' + str(r2)) #给图写上title
#标注
#设置标注,inv间隔   
inv = (plt.axis()[3]-plt.axis()[2])
xmin =plt.axis()[2]


plt.xlabel('地基站点观测值(μg/${m^3}$)')#x轴名称
plt.ylabel('模型预测值(μg/${m^3}$)')#y轴名称
plt.text(min(y_test),xmin + 0.9*inv,r'地区: 全国' )#,fontsize=12
plt.text(min(y_test),xmin + 0.85*inv,'时间: 2020年')
#plt.text(min(x),xmin + 0.8*inv,s='时间: 2019年'+first_month +'-'+ last_month +'月',fontsize=12)
plt.text(min(y_test),xmin + 0.8*inv,r'R2: '+ str(r2)[:6])
plt.text(min(y_test),xmin + 0.75*inv,r'RMSE:'+str(rmse)[:6])
plt.text(min(y_test),xmin + 0.7*inv,r'MAE:'+str(mae)[:6])
plt.text(min(y_test),xmin + 0.65*inv,r'N: '+ str(len(y_test)))




#plt.savefig(r'K:\jpg\ERF\ERF_2020.png')
plt.show()



toc = time.time()
print("用时",(datetime.datetime.fromtimestamp(toc)-datetime.datetime.fromtimestamp(tic)).seconds,"秒")

















