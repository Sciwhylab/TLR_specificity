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
    print('\n Prediction of uncharacterized TLRs: \n', nbt_df)    

    return (nbt_df)
    
    
if __name__=='__main__':
    
    input_file = argv[1]
    #output_dir = argv[2]
    #os.makedirs(output_dir, exist_ok=True)
    
    ## Predicting Novel and Blind set TLR specificity  
    nbdata = pd.read_csv(input_file, skiprows=2, sep="\t")
    #print(nbdata)
    
    my_model = joblib.load('RFC-LOO_model.pkl')
    
    nbt_df = uncharacterize_pred(nbdata, my_model)    
    #nbt_df.to_csv(os.path.join(output_dir, "prediction_output.tsv"), sep="\t") 
        
    
