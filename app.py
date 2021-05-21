# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


from flask import Flask, jsonify
from flask import request
import os
import pickle
import logging
import pandas as pd
app = Flask(__name__)



independent_variables = ['Violation code',
        'Route',
        'Body Style',
        'Agency']


# Initialuze logger
logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
@app.route("/prediction")
def get_predictions():
    """
    This route provides prediction response by loading model and target encoding pickled object.

    Returns
    -------
    JSON
        Response with prediction and status code.

    """
    
    # Get model from pickle 
    file_name = "/".join([os.getcwd(), 'pickle_model', 'model.sav'])
    fileObject = open(file_name,'rb')
    model = pickle.load(fileObject) 
    fileObject.close()
    logger.info("Model object  loaded sucessfully")
    
    # Get target Encoding Details
    file_name = "/".join([os.getcwd(), 'pickle_encoding', 'encode.sav'])
    fileObject = open(file_name,'rb')
    encode_dict = pickle.load(fileObject) 
    fileObject.close()
    logger.info("Target encoding parameters loaded sucessfully")
    result = dict()
    logger.info((request.args))
    for col in independent_variables:
        if col not in request.args:
            logger.info("Independent variable not present in model training")
            return jsonify({
                    "status": "failed",
                    "code": 422})
        
        req_param = request.args.get(col)
        target_dict_encoding = encode_dict.get(col, None)
        if not target_dict_encoding : 
            jsonify({
                    "status": "failed",
                    "code": 422})
            
        for target_col in target_dict_encoding:
            encoded_value = 0
            if req_param in target_dict_encoding[target_col]:
                
                encoded_value = target_dict_encoding[target_col].get(req_param, 0)
            key = "_".join([col.replace(" ", ""), target_col])
            result[key] = [encoded_value]
            
    logger.info(len(result))
    df_pred = pd.DataFrame(result)
    dtree_predictions = model.predict_proba(df_pred)
    logger.info("Prediction Sucessful")
    probability, make = max(zip(dtree_predictions[0], model.classes_),
        key = lambda x : x[0])
    
    return jsonify({make:probability,
                    "status": "Success",
                    "code": 200})
    


if __name__ == "__main__":
    app.run(host='0.0.0.0')