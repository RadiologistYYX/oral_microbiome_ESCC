# Python 3.11.5 
# -*- coding: utf-8 -*-



import pandas as pd
from numpy import interp
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import cloudpickle as pickle


CEA_df = pd.read_csv("Figure7_CEA.csv")
CEA_df = pd.read_csv("Figure7_CEA.csv")
### Due to the data being derived from Chinese serum samples, it is subject to relevant legal and regulatory requirements. If you need access to this data, please contact the corresponding author to obtain limited access to this information.

CEA_df["label"][CEA_df["label"]=="Control"]=0
CEA_df["label"][CEA_df["label"]=="ESCC"]=1

CEA_df1 = CEA_df.dropna()
SCC_df1 = SCC_df.dropna()

X = CEA_df1.drop("label",axis=1)
y = CEA_df1["label"]
X = X.astype('int64')
y = y.astype('int64')

SEED, best_auc, best_features, best_plot_data, feature_rank = pickle.load(open('.data/Test_asvs_cea.pkl', 'rb'))
lines, mean_fpr, mean_tpr, tprs, std_auc = best_plot_data

for train_index,test_index in kf.split(X,y):
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    X_train, X_test = X.iloc[train_index], X.iloc[test_index] 
    y_train, y_test = y[train_index], y[test_index]
    model = RandomForestClassifier(max_depth=best_depth,n_estimators=best_estimator,random_state=SEED) 
    model.fit(X_train, y_train)
    fpr,tpr,thresholds=roc_curve(y_test, model.predict_proba(X_test)[:,1])
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    plt.plot(fpr,tpr,lw=1,alpha=0.3)
    #plt.plot(fpr,tpr,lw=1,alpha=0.3,label='ROC fold %d(area=%0.2f)'% (i,roc_auc))
    roc_auc=auc(fpr,tpr)
    aucs.append(roc_auc)
    i +=1
max_length = max(len(fpr) for fpr in fpr_list)

interpolated_fpr_list = []
interpolated_tpr_list = []

for fpr, tpr in zip(fpr_list, tpr_list):
    interpolated_fpr = np.interp(np.linspace(0, 1, max_length), np.linspace(0, 1, len(fpr)), fpr)
    interpolated_tpr = np.interp(np.linspace(0, 1, max_length), np.linspace(0, 1, len(tpr)), tpr)
    interpolated_fpr_list.append(interpolated_fpr)
    interpolated_tpr_list.append(interpolated_tpr)

mean_fpr = np.mean(interpolated_fpr_list, axis=0)
mean_tpr = np.mean(interpolated_tpr_list, axis=0)

std_tpr = np.std(interpolated_tpr_list, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

mean_auc = auc(mean_fpr, mean_tpr)



std_auc = np.std(aucs)

plt.plot([0, 1], [0, 1], ls='--', lw=2, color='r')
plt.plot(mean_fpr, mean_tpr, color='b', lw=2.5,label='Mean ROC(AUC=%0.2f $\pm$ %0.2f)'%(mean_auc, std_auc))
plt.legend(loc='lower right',prop=font1)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', font2)
plt.ylabel('True Positive Rate', font2)
plt.yticks(fontproperties = 'Arial', size = 13)
plt.xticks(fontproperties = 'Arial', size = 13)
plt.savefig("Figure7C_CEA.pdf")




y_pred = cross_val_predict(model,X,Y,cv=10)
cm = confusion_matrix(Y, y_pred)  
print('Cross predict matrix ：\n',cm)
import matplotlib.pyplot as plt  
plt.subplots(dpi=300,figsize=(5,4),facecolor='White')
plt.matshow(cm, cmap=plt.cm.Greens,fignum=0)  
plt.colorbar()  

for x in range(len(cm)):  
    for y in range(len(cm)):
        plt.annotate(cm[x, y], xy=(y, x),verticalalignment='center',horizontalalignment='center')

plt.ylabel('True label')  
plt.xlabel('Predicted label')  
plt.title('Confusion Matrix of CEA in Chongqing Cohort')
plt.savefig("Figure7D_SCC-Ag.pdf")

plt.show()

X = SCC_df1.drop("label",axis=1)
y = SCC_df1["label"]
X = X.astype('int64')
y = y.astype('int64')



SEED, best_auc, best_features, best_plot_data, feature_rank = pickle.load(open('.data/Test_asvs_scc.pkl', 'rb'))
lines, mean_fpr, mean_tpr, tprs, std_auc = best_plot_data

for train_index,test_index in kf.split(X,y):
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    X_train, X_test = X.iloc[train_index], X.iloc[test_index] 
    y_train, y_test = y[train_index], y[test_index]
    model = RandomForestClassifier(max_depth=best_depth,n_estimators=best_estimator,random_state=SEED) 
    model.fit(X_train, y_train)
    fpr,tpr,thresholds=roc_curve(y_test, model.predict_proba(X_test)[:,1])
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    plt.plot(fpr,tpr,lw=1,alpha=0.3)
    #plt.plot(fpr,tpr,lw=1,alpha=0.3,label='ROC fold %d(area=%0.2f)'% (i,roc_auc))
    roc_auc=auc(fpr,tpr)
    aucs.append(roc_auc)
    i +=1
max_length = max(len(fpr) for fpr in fpr_list)

interpolated_fpr_list = []
interpolated_tpr_list = []

for fpr, tpr in zip(fpr_list, tpr_list):
    interpolated_fpr = np.interp(np.linspace(0, 1, max_length), np.linspace(0, 1, len(fpr)), fpr)
    interpolated_tpr = np.interp(np.linspace(0, 1, max_length), np.linspace(0, 1, len(tpr)), tpr)
    interpolated_fpr_list.append(interpolated_fpr)
    interpolated_tpr_list.append(interpolated_tpr)

mean_fpr = np.mean(interpolated_fpr_list, axis=0)
mean_tpr = np.mean(interpolated_tpr_list, axis=0)

std_tpr = np.std(interpolated_tpr_list, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

mean_auc = auc(mean_fpr, mean_tpr)



std_auc = np.std(aucs)

plt.plot([0, 1], [0, 1], ls='--', lw=2, color='r')
plt.plot(mean_fpr, mean_tpr, color='b', lw=2.5,label='Mean ROC(AUC=%0.2f $\pm$ %0.2f)'%(mean_auc, std_auc))
plt.legend(loc='lower right',prop=font1)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', font2)
plt.ylabel('True Positive Rate', font2)
plt.yticks(fontproperties = 'Arial', size = 13)
plt.xticks(fontproperties = 'Arial', size = 13)
plt.savefig("Figure7C_SCC-Ag.pdf")




y_pred = cross_val_predict(model,X,Y,cv=10)
cm = confusion_matrix(Y, y_pred)  
print('Cross predict matrix ：\n',cm)
import matplotlib.pyplot as plt  
plt.subplots(dpi=300,figsize=(5,4),facecolor='White')
plt.matshow(cm, cmap=plt.cm.Greens,fignum=0)  
plt.colorbar()  

for x in range(len(cm)):  
    for y in range(len(cm)):
        plt.annotate(cm[x, y], xy=(y, x),verticalalignment='center',horizontalalignment='center')

plt.ylabel('True label')  
plt.xlabel('Predicted label')  
plt.title('Confusion Matrix of SCC in Chongqing Cohort')
plt.savefig("Figure7D_SCC-Ag.pdf")

plt.show()





