# Python 3.11.5 
# -*- coding: utf-8 -*-

import pandas as pd

total_df = pd.read_csv("./data/Figure5A.csv")

total_df1 = total_df.drop("batch",axis=1)
total_df1["group"][total_df1["group"]=="Control"]=0
total_df1["group"][total_df1["group"]=="Disease"]=1

X = total_df1.drop("group",axis=1)
y = total_df1["group"]
X = X.astype('int64')
y = y.astype('int64')


from numpy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=14)
i = 1
best_auc = 0
font1 = {'family' : 'Arial','weight' : 'normal','size': 12}
font2 = {'family' : 'Arial','weight' : 'normal','size': 16}
fpr_list = []
tpr_list = []
tprs = []
aucs = []
for train_index,test_index in kf.split(X,y):
    #建立模型，并对训练集进行测试，求出预测得分
    #划分训练集和测试集
    #print('\n{} of kfold {}'.format(i,kf.n_splits))
    X_train, X_test = X.iloc[train_index], X.iloc[test_index] 
    y_train, y_test = y[train_index], y[test_index]
    #建立模型(模型已经定义)
    model = RandomForestClassifier(max_depth=2,n_estimators=26,random_state=630)
    #训练模型
    model.fit(X_train, y_train)
    #计算fpr(假阳性率),tpr(真阳性率),thresholds(阈值)[绘制ROC曲线要用到这几个值]
    fpr,tpr,thresholds=roc_curve(y_test, model.predict_proba(X_test)[:,1])
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    plt.plot(fpr,tpr,lw=1,alpha=0.3)
    #plt.plot(fpr,tpr,lw=1,alpha=0.3,label='ROC fold %d(area=%0.2f)'% (i,roc_auc))
    roc_auc=auc(fpr,tpr)
    aucs.append(roc_auc)
    i +=1
# 最大长度
max_length = max(len(fpr) for fpr in fpr_list)

# 插值填充数据点
interpolated_fpr_list = []
interpolated_tpr_list = []

for fpr, tpr in zip(fpr_list, tpr_list):
    interpolated_fpr = np.interp(np.linspace(0, 1, max_length), np.linspace(0, 1, len(fpr)), fpr)
    interpolated_tpr = np.interp(np.linspace(0, 1, max_length), np.linspace(0, 1, len(tpr)), tpr)
    interpolated_fpr_list.append(interpolated_fpr)
    interpolated_tpr_list.append(interpolated_tpr)

# 计算平均FPR和TPR
mean_fpr = np.mean(interpolated_fpr_list, axis=0)
mean_tpr = np.mean(interpolated_tpr_list, axis=0)
# 灰度区间
std_tpr = np.std(interpolated_tpr_list, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')
# 计算平均的AUC
mean_auc = auc(mean_fpr, mean_tpr)

#计算auc
#roc_auc=auc(fpr,tpr)
#aucs.append(roc_auc)
std_auc = np.std(aucs)
#if(best_auc<roc_auc):
#    best_auc = roc_auc
#画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数计算出来
plt.plot([0, 1], [0, 1], ls='--', lw=2, color='r')
plt.plot(mean_fpr, mean_tpr, color='b', lw=2.5,label='Mean ROC(AUC=%0.2f $\pm$ %0.2f)'%(mean_auc, std_auc))
plt.legend(loc='lower right',prop=font1)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', font2)
plt.ylabel('True Positive Rate', font2)
plt.yticks(fontproperties = 'Arial', size = 13)
plt.xticks(fontproperties = 'Arial', size = 13)
plt.savefig("Figure5A.pdf")
