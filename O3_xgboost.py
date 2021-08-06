# -*- coding: utf-8 -*-
#GBDT model processing: https://blog.csdn.net/duanlianvip/article/details/102398886

import time
import datetime
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import GridSearchCV  #https://blog.csdn.net/weixin_41988628/article/details/83098130  参数说明
from sklearn import linear_model 
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
import matplotlib.colors as colors
from xgboost import XGBRegressor


pd_data = pd.read_csv(r'K:\csv\2020_train_delete.csv',encoding='gb2312')





tic = time.time()
#,'lon','lat',

X = pd_data.loc[:, ('month','day','lon','lat',
                    'CO','HCHO','NO2','AOD',
                    'vieo','vino',
                    'sp','rh','blh','ssrd','t2m','u10','v10','cbh','alnid')]

#X = pd_data.loc[:, ('month','day','lat','lon','blh','r','sp','ssrd','t2m','tcc','tp','u10','v10','CO','HCHO','NO2','SO2')]

y = pd_data.loc[:, 'O3']
#y = pd_data.loc[:, 'pm25']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

'''
axisx = np.arange(0.2,0.3,0.01)
ScoreAll = []
#for i in range(5,15,1):
for i in axisx:   
    clf_e = XGBRegressor(n_estimators=300,max_depth=9,learning_rate=i)#0.15,
    clf_e.fit(X_train, y_train)
    
    model_result = pd.Series(clf_e.predict(X_test))
    score = r2_score(y_test,model_result)
    
    print('min_samples_split :',i,'R2: %.4f'%score)
    ScoreAll.append([i,score])
   
ScoreAll = np.array(ScoreAll)
max_score = np.where(ScoreAll==np.max(ScoreAll[:,1]))[0][0] ##这句话看似很长的，其实就是找出最高得分对应的索引
print("最优参数以及最高得分:",ScoreAll[max_score])  
plt.figure(figsize=[20,5])
plt.plot(ScoreAll[:,0],ScoreAll[:,1])
plt.show()


'''
model = XGBRegressor(n_estimators=500,max_depth=10,learning_rate=0.1)
# delete 最优参数#n_estimators=500,max_depth=10,
model.fit(X_train,y_train)



#导出模型
#with open(r'K:\code\model\2019_xgboost_net.pickle','wb') as fw:
    #pickle.dump(model,fw)


# 使用模型预测
y_predict = model.predict(X_test)
y_predict = pd.Series(y_predict)

#导出模型
with open(r'K:\model\xgboost\2020_xgboost.pickle','wb') as fw:
    pickle.dump(model,fw)
#本来是乱序重新索引
y_test=y_test.reset_index(drop=True)



corr = round(y_test.corr(y_predict),4)
r2 = round((corr*corr),4)
rmse = np.sqrt(mean_squared_error(y_test,y_predict))
mae = mean_absolute_error(y_test,y_predict)


from scipy.stats import gaussian_kde
# Calculate the point density
xy = np.vstack([y_test,y_predict])
z = gaussian_kde(xy)(xy)



plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False

fig,ax  = plt.subplots()

density = plt.scatter(y_test, y_predict, c=z, s=0.1,vmin=0.00001,vmax=0.00015)#norm=colors.LogNorm(vmin=0.00005,vmax=0.0003)s=0.1,vmin=0.00001,vmax=0.0003


fig.colorbar(density,label='高斯核密度')


a,b = np.polyfit(y_test,y_predict,1)
plt.plot([0,500],[b,a*500+b],color='blue',linestyle='--')#拟合线
plt.plot([0,500],[0,500],color='red',linestyle='--')# y=x线

#plt.title('r2:'+ str(r2)) #给图写上title
#标注
#设置标注,inv间隔   
inv = (plt.axis()[3]-plt.axis()[2])
xmin =plt.axis()[2]


plt.xlabel('地基站点观测值(μg/${m^3}$)')#x轴名称
plt.ylabel('模型预测值(μg/${m^3}$)')#y轴名称
plt.text(min(y_test),xmin + 0.9*inv,r'地区: 全国' )#,fontsize=12
plt.text(min(y_test),xmin + 0.85*inv,'时间: 2020年')
#plt.text(min(x),xmin + 0.8*inv,s='时间: 2019年'+first_month +'-'+ last_month +'月',fontsize=12)
plt.text(min(y_test),xmin + 0.8*inv,r'R2: '+ str(r2))
plt.text(min(y_test),xmin + 0.75*inv,r'RMSE:'+str(rmse)[:6])
plt.text(min(y_test),xmin + 0.7*inv,r'MAE:'+str(mae)[:6])
plt.text(min(y_test),xmin + 0.65*inv,r'N: '+ str(len(y_test)))


plt.savefig(r'K:\jpg\xgboost\2020_xgboost.png')
plt.show()



toc = time.time()
print("用时",(datetime.datetime.fromtimestamp(toc)-datetime.datetime.fromtimestamp(tic)).seconds,"秒")



