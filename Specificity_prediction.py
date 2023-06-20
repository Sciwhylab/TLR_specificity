#!/usr/bin/env python
# coding: utf-8


import os
from sys import argv
import pandas as pd
import joblib


def uncharacterize_pred(nbdata, my_model):
    
    ## Prediction for uncharacterized TLRs
    rdata = nbdata.iloc[:,1:28]
    
    pred_classes = []
    for index, row in rdata.iterrows():
        #print(row)
        pred_class = my_model.predict([row])
        pred_classes.append(pred_class[0])
    #print(pred_classes)

    nbdata['Predicted_class'] = pd.Series(pred_classes)
    nbt_df = nbdata[['Gene_name', 'Predicted_class']]
    #print('\n Prediction of uncharacterized TLRs: \n', nbt_df)    

    return (nbt_df)
    
    
if __name__=='__main__':
    
    input_file = argv[1]
    
    ## Predicting Novel and Blind set TLR specificity  
    nbdata = pd.read_csv(input_file, skiprows=2, sep="\t")
    #print(nbdata)
    
    my_model = joblib.load('Src/RFC-LOO_model.pkl')
    
    nbt_df = uncharacterize_pred(nbdata, my_model)    
    nbt_df.to_csv("TLR_specificity_prediction.tsv", sep="\t", index=None) 
        
    
