#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sys import argv
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import matthews_corrcoef, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc



def data_preprocessing(mdata):
    
    ## Assign interger labels to tissue names 
    le = preprocessing.LabelEncoder()
    le.fit(mdata["Specific tissue"])
    st = le.transform(mdata["Specific tissue"])
    ST = list(st)
    mdata["Specific tissue"] = ST

    ## Define strand to pathogen nucleic acids
    Dict = {'dsRNA':'ds', 'ssRNA':'ss', 'ssDNA':'ss', 'Other_TLRs':'other'}
    mdata['Group_strand_specific'] = mdata['Group_new'].map(Dict)
    #print(mdata)

    ## Remove extra columns 
    data = mdata.drop(['Ensembl_id','Gene_name','Species'], axis = 1)
    G4data = data.drop(['Group_based_on_NA_sensing','Type_of_sensed_NA','Group_new'], axis = 1)
    #print(G4data)

    ## Separate target labels from input data 
    X = G4data.iloc[:,0:27]
    Y = G4data.iloc[:,-1]
    return (X, Y)


def rfc_loo(X,Y):
    
    ## Create leave-one-out cross validation
    cv = LeaveOneOut()
    Y_true, Y_pred, Y_pred_proba = list(), list(), list()

    for train_ix, test_ix in cv.split(X):
        
        ## Split the data into train and test
        X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
        Y_train, Y_test = Y.iloc[train_ix], Y.iloc[test_ix]
        #print(Y_train, Y_test)
        
        ## Fit the model on train data and hyperparameter tuning
        model = RandomForestClassifier(n_estimators=100, random_state=1)
        #model = RandomForestClassifier(n_estimators=100, random_state=1, class_weight="balanced_subsample")
        model.fit(X_train, Y_train)

        ## Predict on test data
        yhat = model.predict(X_test)
        yhat_proba = model.predict_proba(X_test)

        Y_true.append(Y_test.tolist())
        Y_pred.append(yhat[0])
        Y_pred_proba.append(yhat_proba[0])
        
    return (Y_true, Y_pred, Y_pred_proba, model)
    
    
def overall_performance(Y_true, Y_pred):
    
    # Measure accuracy of model
    acc = accuracy_score(Y_true, Y_pred)
    print('\n Overall Accuracy of RFC-LOO model: %.3f' % acc)

    Y_list = [''.join(ele) for ele in Y_true]
    Y_true_arr = pd.Series(Y_list)
    Y_pred_arr = np.array(Y_pred)

    # Measure Matthews correlation coefficient of model
    mcc = matthews_corrcoef(Y_true_arr, Y_pred_arr)
    print('\n Overall Matthews correlation coefficient of RFC-LOO model: %.3f' % mcc)
    
    return (Y_true_arr, Y_pred_arr, acc, mcc)
    
    
def each_class_performance(Y_true_arr, Y_pred_arr):
    
    ## Making confusion matrix
    cm1 = confusion_matrix(Y_pred_arr, Y_true_arr)
    print('\n Confusion Matrix: \n', cm1)

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
    TPR = np.array([round(i, 3) for i in TPR])

    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    TNR = np.array([round(i, 3) for i in TNR])

    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    PPV = np.array([round(i, 3) for i in PPV])

    # Negative predictive value
    NPV = TN/(TN+FN)
    NPV = np.array([round(i, 3) for i in NPV])

    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    FPR = np.array([round(i, 3) for i in FPR])

    # False negative rate
    FNR = FN/(TP+FN)
    FNR = np.array([round(i, 3) for i in FNR])

    # False discovery rate
    FDR = FP/(TP+FP)
    FDR = np.array([round(i, 3) for i in FDR])

    # Overall accuracy for each class
    ACC = (TP+TN)/(TP+FP+FN+TN)
    ACC = np.array([round(i, 3) for i in ACC])

    # Matthew’s correlation coefficient
    MCC = ((TP*TN)-(FP*FN))/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    MCC = np.array([round(i, 3) for i in MCC])

    arr = np.vstack((PPV, TPR, TNR, NPV, FPR, FNR, FDR, ACC, MCC))
    column_values = ['dsNA-sensing TLRs','Other TLRs','ssNA-sensing TLRs']
    index_values = ['PPV/Precision', 'TPR/Sensitivity/Recall', 'TNR/Specificity', 'NPV', 'FPR', 'FNR', 'FDR', 'ACC', 'MCC']
    df = pd.DataFrame(data = arr, index = index_values, columns = column_values)
    ecp_df = df.T
    print('\n Each class performance measures: \n', ecp_df)

    return (ecp_df)


def misclassified_pred(Y_true_arr, Y_pred_arr, mdata):
    
    ## Finding misclassified prediction
    mistakes = np.invert(Y_true_arr == Y_pred_arr)

    X_test_mis = mdata[mistakes]
    X_test_mis = X_test_mis.drop(X_test_mis.iloc[:, 3:33], axis=1)
    X_test_mis.rename(columns = {'Group_strand_specific':'True_class'}, inplace = True)
    mis_df = X_test_mis.reset_index(drop=True)
    
    ## False negative predictions
    Y_pred_FN = Y_pred_arr[mistakes]
    Y_pred_FN = pd.Series(Y_pred_FN)
    
    mis_df['Predicted_class'] = Y_pred_FN
    print('\n Misclassified predictions: \n', mis_df)
        
    return (mis_df)
    
    
def measure_auc(Y_true, Y_pred, Y_pred_proba):
    
    ## Binarize the RFC-LOO model output
    Y_test_bin = label_binarize(Y_true, classes=['ds', 'other', 'ss'])
    Y_pred_bin = label_binarize(Y_pred, classes=['ds', 'other', 'ss'])
    n_classes = Y_pred_bin.shape[1]

    Y_pred_proba = np.array(Y_pred_proba)

    ## Compute false and true positive rate for different classification thresholds
    ## Get area under the curve score
    n_classes = 3
    fpr = [0] * 3
    tpr = [0] * 3
    thresholds = [0] * 3
    auc_score = [0] * 3

    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(Y_test_bin[:, i], Y_pred_proba[:, i])
        auc_score[i] = auc(fpr[i], tpr[i])
    #print(auc_score)
    
    return (fpr, tpr, auc_score)



if __name__=='__main__':
    
    input_path1 = argv[1]
    output_dir = argv[2]
    os.makedirs(output_dir, exist_ok=True)
    
    
    ## Load training set TLRs
    mdata = pd.read_csv(input_path1, sep="\t")
    #print(mdata)
    
    
    ## Preprocessing of input data
    X, Y = data_preprocessing(mdata)
    
    
    ## Establish and save the RFC-LOO model
    Y_true, Y_pred, Y_pred_proba, model = rfc_loo(X,Y)
    #print(Y_true, Y_pred, Y_pred_proba)
    joblib.dump(model, 'Src/RFC-LOO_model.pkl')

    
    ## Measure overall performance of RFC-LOO model
    Y_true_arr, Y_pred_arr, acc, mcc = overall_performance(Y_true, Y_pred)
    out_file = open("Doc/RFC-LOO_model_Overall_performance.txt", "a")
    out_file.write('Overall Accuracy of RFC-LOO model: ')
    out_file.write(str(round(acc,3)))
    out_file.write('\nOverall Matthews correlation coefficient of RFC-LOO model: ')
    out_file.write(str(round(mcc,3)))
    out_file.close()
    
    
    ## Measure each class performance of RFC-LOO model
    ecp_df = each_class_performance(Y_true_arr, Y_pred_arr)    
    ecp_df.to_csv(os.path.join(output_dir, "RFC-LOO_model_Each_class_performance.tsv"), sep="\t")
    
    
    ## Predict the misclassified TLRs
    mis_df = misclassified_pred(Y_true_arr, Y_pred_arr, mdata)
    mis_df.to_csv(os.path.join(output_dir, "RFC-LOO_model_Misclassified_predictions.tsv"), sep="\t", index=None)
    
        
    ## Compute the ROC AUC score for each class     
    fpr, tpr, auc_score = measure_auc(Y_true, Y_pred, Y_pred_proba)
    
    ## Plot ROC curve for multi class classification
    plt.figure(figsize=(6,4))
    plt.plot(fpr[0], tpr[0], linestyle='--',color='orange', label='dsNA-sensing TLRs (AUC='+str(round(auc_score[0], 3))+')')
    plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label='ssNA-sensing TLRs (AUC='+str(round(auc_score[2], 3))+')')
    plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label='Other TLRs (AUC='+str(round(auc_score[1], 3))+')')
    plt.title('RFC LOO Model: ROC-Curve', weight='bold', fontsize=16, fontname='Times New Roman')
    plt.xlabel('Sensitivity', weight='bold', fontsize=12, fontname='Times New Roman')
    plt.ylabel('1-Specificity', weight='bold', fontsize=12, fontname='Times New Roman')
    plt.xticks(fontsize=10, weight='bold', fontname='Times New Roman')
    plt.yticks(fontsize=10, weight='bold', fontname='Times New Roman')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right', prop={'size': 10, 'weight':'bold'})
    plt.savefig(os.path.join(output_dir, 'RFC-LOO_Model_ROC_Curve.png'), dpi=500); 
   

    ## Retrieve the important features with RFC-LOO model
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]

    ## Plot the importance features of TLRs
    f, ax = plt.subplots(figsize=(12,6))
    plt.bar(range(X.shape[1]), importances[sorted_indices], align='center',color=['blue','purple','red','green','magenta','yellow','cyan','tan'])
    plt.xticks(range(X.shape[1]), X.columns[sorted_indices], rotation=90, fontsize=12, weight='bold', fontname='Times New Roman')
    plt.yticks(fontsize=10, weight='bold', fontname='Times New Roman')
    plt.xlabel('Number of Features', fontsize=14, weight='bold', fontname='Times New Roman')
    plt.ylabel('Feature Importance', fontsize=14, weight='bold', fontname='Times New Roman')
    plt.tight_layout()
    plt.show()
    f.savefig(os.path.join(output_dir,'RFC-LOO_Model_Important_features.png'), dpi=500)
       
    
