#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc



## load input data
mdata = pd.read_csv("Final_ML_27_recent_features.txt", sep="\t")
#print(mdata)


## Assign interger labels to tissue names  
le = preprocessing.LabelEncoder()
le.fit(mdata["Specific tissue"])
st = le.transform(mdata["Specific tissue"])
ST = list(st)
mdata["Specific tissue"] = ST

Dict = {'dsRNA':'ds', 'ssRNA':'ss', 'ssDNA':'ss', 'Other_TLRs':'other'}
mdata['Group_strand_specific'] = mdata['Group_new'].map(Dict)
#print(mdata)


## Remove extra columns 
data = mdata.drop(['Ensembl_id','Gene_name','Species'], axis = 1)
G4data = data.drop(['Group_based_on_NA_sensing','Type_of_sensed_NA','Group_new'], axis = 1)
print(G4data)


## Separate target labels from input data 
X = G4data.iloc[:,0:27]
Y = G4data.iloc[:,-1]


## Run RFC-LOO model
cv = LeaveOneOut()
Y_true, Y_pred, Y_pred_proba = list(), list(), list()

for train_ix, test_ix in cv.split(X):
    #print(len(test_ix))
    
    X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
    Y_train, Y_test = Y.iloc[train_ix], Y.iloc[test_ix]
    #print(Y_train, Y_test)
       
    model = RandomForestClassifier(n_estimators=100, random_state=1)
    #model = RandomForestClassifier(n_estimators=100, random_state=1, class_weight="balanced_subsample")
    model.fit(X_train, Y_train)
    
    yhat = model.predict(X_test)
    yhat_proba = model.predict_proba(X_test)
    
    Y_true.append(Y_test.tolist())
    Y_pred.append(yhat[0])
    Y_pred_proba.append(yhat_proba[0])



## Evaluate performance metrics of RFC-LOO model     
acc = accuracy_score(Y_true, Y_pred)
print('Overall Accuracy of RFC-LOO model: %.3f' % acc)

Y_list = [''.join(ele) for ele in Y_true]
Y_seri = pd.Series(Y_list)
Y_pred_arr = np.array(Y_pred)

mcc = matthews_corrcoef(Y_pred_arr, Y_seri)
print('Overall Matthews correlation coefficient of RFC-LOO model: %.3f' % mcc)



## Other performance metrics of RFC-LOO model for each class     
cm1 = confusion_matrix(Y_pred_arr, Y_seri)
print('Confusion Matrix: \n', cm1)

FP = cm1.sum(axis=0) - np.diag(cm1) 
FN = cm1.sum(axis=1) - np.diag(cm1)
TP = np.diag(cm1)
TN = cm1.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)

# Specificity or true negative rate
TNR = TN/(TN+FP) 

# Precision or positive predictive value
PPV = TP/(TP+FP)

# Negative predictive value
NPV = TN/(TN+FN)

# Fall out or false positive rate
FPR = FP/(FP+TN)

# False negative rate
FNR = FN/(TP+FN)

# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy for each class
ACC = (TP+TN)/(TP+FP+FN+TN)

# Matthew’s correlation coefficient
MCC = ((TP*TN)-(FP*FN))/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))


arr = np.vstack((PPV, TPR, TNR, NPV, FPR, FNR, FDR, ACC, MCC))
column_values = ['dsNA-sensing TLRs','Other TLRs','ssNA-sensing TLRs']
index_values = ['PPV/Precision', 'TPR/Sensitivity/Recall', 'TNR/Specificity', 'NPV', 'FPR', 'FNR', 'FDR', 'ACC', 'MCC']
df = pd.DataFrame(data = arr, index = index_values, columns = column_values)
dff = df.T
dff.to_csv("RFC-LOO_performance_measures.tsv", sep="\t")
print('Other performance measures: \n', dff)



## Identification of misclassified predictions
mistakes = np.invert(Y_pred_arr == Y_seri)

X_test_mis = X[mistakes]
print('Misclassified predictions: \n', X_test_mis)

Y_test_mis = Y[mistakes]
print(Y_test_mis)

Y_pred_FN = Y_pred_arr[mistakes]
print(Y_pred_FN)

## Seven are misclassified predictions:
# 25 : Zebrafish TLR19 : ds -> ss
# 28 : Zebrafish TLR22 : ds -> ss
# 41 : Chicken TLR15 : ss -> other
# 65 : Opossum TLR13 : ss -> ds
# 69 : Ferret TLR3 : ds -> other
# 117 : Pig TLR3 : ds -> other 
# 120 : Frog TLR13 : ss -> ds    



## Novel TLRs
sTLR18=[48.69,3.78024528386535,2.90741136077459,645,0.642289976809729,1,1.32896371373046,13.8614,3,5,3,151,0,0,0,1,1,0,1,3,0,1,2,5,0.866336633663366,29.5139,33.6806]
sTLR25a=[39.74,3.37529773821734,2.89762709129044,411,0.678600069791113,1,0.53327996080596,11.2516,3,4,2,117,0,0,0,1,1,0,1,3,0,1,2,5,0.379266750948167,28.5968,33.3925]
sTLR25b=[40.42,3.39776625612645,2.92012332629072,411,0.56875518822578,1,2.25588311670098,10.2163,3,4,2,76,0,0,0,1,1,0,1,3,0,1,2,5,0.480769230769231,28,33.913]
nTLR25=[39.91,3.7041505168398,2.90091306773767,411,0.936958255720337,2,0.847488236883368,12.0603,2,2,2,116,0,0,0,1,1,0,1,2,0,1,2,4,0.628140703517588,28.9425,34.3228]
sTLR27=[41.05,3.37254380075907,2.89486965674525,645,0.941836032202795,7,0.082540644740307,12.4682,4,4,2,134,0,0,0,1,1,0,1,4,0,1,2,6,0.381679389312977,27.3852,34.0989]

## Blind TLRS
aTLR9=[43.83,3.79574102086924,3.03059972196595,544,1,7,0.006288073603163,7.2897,2,3,2,111,0,0,0,0,1,0,0,1,0,1,1,2,1.21155638397018,31.4149,33.0935]
aTLR18=[48.18,3.45591024038274,2.96894968098134,645,0.548810503468758,0,1.18466188731694,9.4828,3,4,2,106,0,0,0,0,1,0,0,1,0,1,1,2,5.15574650912997,26.6436,34.2561]

## Prediction for novel and blind TLRs
new_data = [sTLR18,sTLR25a,sTLR25b,nTLR25,sTLR27, aTLR9,aTLR18]
for ind, val in enumerate(new_data):
    #print(ind)
    pred_class = model.predict([new_data[ind]])
    print(str(ind) + "\t" + pred_class[0])
    


## Calculate AUC and ROC for multiclass RFC-LOO model
Y_test_bin = label_binarize(Y_true, classes=['ds', 'other', 'ss'])
Y_pred_bin = label_binarize(Y_pred, classes=['ds', 'other', 'ss'])
n_classes = Y_pred_bin.shape[1]

Y_pred_proba = np.array(Y_pred_proba)
#print(Y_pred_proba)


# Measure AUC
n_classes = 3
fpr = [0] * 3
tpr = [0] * 3
thresholds = [0] * 3
auc_score = [0] * 3
 
for i in range(n_classes):
    fpr[i], tpr[i], thresholds[i] = roc_curve(Y_test_bin[:, i], Y_pred_proba[:, i])
    auc_score[i] = auc(fpr[i], tpr[i])
print(auc_score)


# plot ROC 
plt.figure(figsize=(6,4))
plt.plot(fpr[0], tpr[0], linestyle='--',color='orange', label='dsNA-sensing TLRs (AUC=0.973)')
plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label='ssNA-sensing TLRs (AUC=0.992)')
plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label='Other TLRs (AUC=0.995)')
plt.title('RFC LOO Model: ROC-Curve', weight='bold', fontsize=16, fontname='Times New Roman')
plt.xlabel('Sensitivity', weight='bold', fontsize=12, fontname='Times New Roman')
plt.ylabel('1-Specificity', weight='bold', fontsize=12, fontname='Times New Roman')
plt.xticks(fontsize=10, weight='bold', fontname='Times New Roman')
plt.yticks(fontsize=10, weight='bold', fontname='Times New Roman')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower right', prop={'size': 10, 'weight':'bold'})
plt.savefig('RFC-LOO_Model_ROC_Curve',dpi=500); 



## Retrieve important features
importances = model.feature_importances_
sorted_indices = np.argsort(importances)[::-1]

f, ax = plt.subplots(figsize=(12,6))
plt.bar(range(X.shape[1]), importances[sorted_indices], align='center',color=['blue','purple','red','green','magenta','yellow','cyan','tan'])
plt.xticks(range(X.shape[1]), X.columns[sorted_indices], rotation=90, fontsize=12, weight='bold', fontname='Times New Roman')
plt.yticks(fontsize=10, weight='bold', fontname='Times New Roman')
plt.xlabel('Number of Features', fontsize=14, weight='bold', fontname='Times New Roman')
plt.ylabel('Feature Importance', fontsize=14, weight='bold', fontname='Times New Roman')
plt.tight_layout()
plt.show()
f.savefig('Feature Importance.png',dpi=300)


