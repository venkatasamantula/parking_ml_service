# -*- coding: utf-8 -*-
"""
Created on Tue May 18 16:21:23 2021

@author: sasha
"""
    
import pickle
import os
import logging
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


logger = logging.getLogger(__name__)


logging.basicConfig(level=logging.INFO)

url = "https://s3-us-west-2.amazonaws.com/pcadsassessment/parking_citations.corrupted.csv"

df = pd.read_csv(url)

path = r"C:\Users\sasha\docker\data\data.csv"
df.to_csv(path, index = False)

# Number of tickets
logger.info(len(set(df['Ticket number'])))

# Ticket column is unique - 8726014


# Seperate Corrupted and uncorrupted data
df_corrupted_data = df[df.Make.isna()]

df_uncorrupted_data = df[~df.Make.isna()]

# Drop columns with more than 10 % nulls
for col in df_uncorrupted_data.columns:

    perc_null_vlaues = df_uncorrupted_data[col].isna().sum() / df_uncorrupted_data.shape[0]
    
    if perc_null_vlaues * 100 > 10:
        logger.info(f"Dropping {col} due to high null values ")
        logger.info((col, perc_null_vlaues) )
        df_uncorrupted_data.drop(columns = [col], 
                                 inplace = True)





# Corrupted records
df_uncorrupted_data['Make'].isna().sum() # 4368470


# Top 25 Make

df_make_freq = df_uncorrupted_data['Make'].value_counts()

top_vehl_makes = set(df_make_freq[:25].index)



# Rename rest of the make to other

df_uncorrupted_data['Make'] = df_uncorrupted_data[['Make']].apply(lambda x : x[0] if x[0] in top_vehl_makes else 'OTHER' , axis = 1)


df_uncorrupted_data['Make'].value_counts()



# importing the required function
def get_correlation_categorical_columns(X, Y, df_stat):
    CrosstabResult=pd.crosstab(index=df_stat[X],
                               columns=df_stat[Y])
 
    # P-Value is the Probability of H0 being True
    # If P-Value>0.05 then only we Accept the assumption(H0)
    # Performing Chi-sq test
    chisq_stat = chi2_contingency(CrosstabResult)
    
    return chisq_stat[1]

 



# Check for multicollinearity
categorical_cols  = [col for col in df_uncorrupted_data.select_dtypes(include = ['object']) if col not in ['Make', 'Ticket number', 'Issue Date']]

# Sampling data due huge data processing
df_sample_corr = df_uncorrupted_data.sample(100000).reset_index(drop = True)


# If p value close to 0 indicate high correlation
for col in categorical_cols:
    logger.info(col)

    p_value = get_correlation_categorical_columns(col,
                                                  'Make',
                                                df_sample_corr)
    logger.info(p_value)
  

# Dtypes
df_uncorrupted_data.dtypes

# Convert to Datetime format
df_uncorrupted_data['Issue Date'] = pd.to_datetime(df_uncorrupted_data['Issue Date'])

df_uncorrupted_data['issue_year_month'] = df_uncorrupted_data[['Issue Date']].apply(lambda x : int(str(x[0].year)+str(x[0].month)), axis = 1)




df_sample = df_uncorrupted_data.sample(100)

"""
# Considering variable that are  highly correlated with target variable from chi square statistic
    
    - Violation Description or Violation code because they give same information
    - Route
    - State
    - Color  # Color might not affect parking violations  by business understanding 
    - Body Style
"""



x_cols = ['Violation code',
        'Route',
        'Body Style',
        'Agency']

y_cols = 'Make'



df_sample = df_uncorrupted_data.sample(100000)

X = df_sample[x_cols]
y = df_sample[y_cols]


# Fill na with mode most frequent occuring

for col in X.columns:
    
    X[col].fillna(X[col].mode().values[0], inplace = True)


y_dummy = pd.get_dummies(y)


# concat y encoded and data
X_encode = pd.concat([X, y_dummy], axis = 1)

target_encoding_map = dict()
for col in x_cols:
    
    for target_col in y_dummy.columns:
        
        means = X_encode.groupby(col)[target_col].mean()
        new_col = "_".join([col.replace(" ", ""), target_col])
        if col not in target_encoding_map:
            target_encoding_map[col] = {target_col: means.to_dict()}
        else:
            target_encoding_map[col].update({target_col: means.to_dict()})
        X_encode[new_col] = X_encode[col].map(target_encoding_map[col][target_col])
        
        
        
X_new_encode = X_encode[target_encoding_map.keys()]       
        
df_sample = X_new_encode.sample(10) 


# Divide the dataset

X_train, X_test, y_train, y_test = train_test_split(X_new_encode, 
                                                    y, 
                                                    test_size=0.33)

    
# training a DescisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train)
dtree_predictions = dtree_model.predict_proba(X_test)

df_predictions = pd.DataFrame(dtree_predictions, columns = dtree_model.classes_)

result = list()
for index, tup in df_predictions.iterrows():
    make, prob = max(tup.to_dict().items(), key = lambda x : x[1])
    result.append([make, prob])
    

df_final = pd.concat([df_predictions, 
                      pd.DataFrame(result, columns= ['make', 'prob'])], axis = 1)




df_sample = df_final.sample(100)

# creating a confusion matrix
cm = confusion_matrix(y_test, dtree_predictions)


# Accuracy Score
accuracy_score(y_test, dtree_predictions, normalize=True)




# Save Model and target_encoding object to pickle  

cwd = os.getcwd()

file_name = "model.sav"

full_path = "\\".join([cwd, 'pickle_model', file_name])
# open the file for writing
fileObject = open(full_path,'wb') 

# this writes the object a to the
# file named 'testfile'
pickle.dump(dtree_model,
            fileObject)   

# here we close the fileObject
fileObject.close()


# Save target encoding parameters to file
file_name = "encode.sav"

full_path = "\\".join([cwd, 'pickle_encoding', file_name])
# open the file for writing
fileObject = open(full_path,'wb') 

# this writes the object a to the
# file named 'testfile'
pickle.dump(target_encoding_map,
            fileObject)   

# here we close the fileObject
fileObject.close()











