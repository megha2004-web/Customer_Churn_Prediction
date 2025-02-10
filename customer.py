#!/usr/bin/env python
# coding: utf-8

# # data loading and understand

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


# load the data
df = pd.read_csv('Customer_Churn.csv')


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


pd.set_option("display.max_columns",None)


# In[6]:


df.head(2)


# In[7]:


df.info()


# In[8]:


# dropping customer_id column as this is not required for modelling
df1 = df.drop(columns = ["customerID"])


# In[9]:


df1.head(2)


# In[10]:


df.columns


# In[11]:


print((df["SeniorCitizen"].unique()))


# In[12]:


# printing the unique values in all the cloumns
for col in df1.columns:
    print(col,df1[col].unique())


# In[13]:


# printing the unique values in all the cloumns(only categrical columns)
numerical_features_list = ["tenure","MonthlyCharges","TotalCharges"]
for col in df1.columns:
    if col not in numerical_features_list:
        print(col,df1[col].unique())


# In[14]:


df1.isnull().sum()


# In[15]:


#df1["TotalCharges"] = df1["TotalCharges"].astype(float)
# here not convert to numerical because of spaces are not convert to numeric


# In[16]:


df1[df1["TotalCharges"]==" "]


# In[17]:


len(df1[df1["TotalCharges"]==" "])


# In[18]:


df1["TotalCharges"] = df1["TotalCharges"].replace({" ":"0.0"})
# here convert space replace the 0.0 


# In[19]:


# convert categrical to numeric
df1["TotalCharges"] = df1["TotalCharges"].astype(float)


# In[20]:


df1.info()


# In[21]:


# checking the class distrubation of target column
print(df1["Churn"].value_counts())


# # Insights
# - customer ID as it is not required for modelling.
# - No missing values in the dataset.
# - Missing values in the Totalcharges column were replaceed with 0.
# - Class inbalance identified in the target.

# # Exploratory Data Analysis(EDA)

# In[22]:


df1.shape


# In[23]:


df1.columns


# In[24]:


df1.head(2)


# In[25]:


df1.describe() # it only worked on numerical columns


# ### Numerical Features : Analysis

# In[26]:


# 1. Understand the distribution of the numerical features


# In[27]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[28]:


def plot_histogram(df1,column_name):
    
    plt.figure(figsize = (5,3))
    sns.histplot(df1[column_name],kde=True)
    plt.title(f"Distribution of {column_name}")
    
    #calculate the mean and median values for the columns
    col_mean = df1[column_name].mean()
    col_median = df1[column_name].median()
    
    # add vertical lines for mean and median
    plt.axvline(col_mean ,color = "red",linestyle = "--",label = "Mean")
    plt.axvline(col_median ,color = "green",linestyle = "--",label = "Median")
    
    plt.legend()
    
    plt.show()


# In[29]:


plot_histogram(df1,"tenure")


# In[30]:


plot_histogram(df1,"MonthlyCharges")


# In[31]:


plot_histogram(df1,"TotalCharges")


# In[32]:


#Box plot for numerical features


# In[33]:


def plot_boxplot(df1,column_name):
    
    plt.figure(figsize = (5,3))
    sns.boxplot(y = df1[column_name])
    plt.title(f"boxplot of {column_name}")
    plt.ylabel(column_name)
    plt.show()


# In[34]:


plot_boxplot(df1,"tenure")


# In[35]:


plot_boxplot(df1,"MonthlyCharges")


# In[36]:


plot_boxplot(df1,"TotalCharges")


# In[37]:


# correlation heatmap for numerical columns


# In[38]:


#correlation matrix --heatmap
plt.figure(figsize = (8,4))
sns.heatmap(df1[["tenure","MonthlyCharges","TotalCharges"]].corr(),annot=True,fmt='.2f',cmap='RdYlGn')
plt.title("Correlation HeatMap")
plt.show()


# ## categorical features - Analysis

# In[39]:


df1.columns


# In[40]:


df1.info()


# In[41]:


# count plot for categorical columns


# In[42]:


obj_cols = df1.select_dtypes(include= "object").columns.to_list()
obj_cols = ["SeniorCitizen"] + obj_cols
obj_cols


# In[43]:


plt.figure(figsize=(20,20))
for wind,col in enumerate(obj_cols, start=1):
    plt.subplot(9,3,wind)
    sns.countplot(x=df1[col])
    plt.title(f"Count plot of {col}")
plt.tight_layout()


# # Data Processing

# In[44]:


df1.head(3)


# In[45]:


# Label encoding of target columns


# In[46]:


from sklearn.preprocessing import LabelEncoder


# In[47]:


df1["Churn"] = df1["Churn"].replace({"Yes":1, "No":0})


# In[48]:


df1.head(3)


# In[49]:


df1["Churn"].value_counts()


# #### LabelEncoding for categorical columns

# In[50]:


# identifying columns with object data type


# In[51]:


object_columns = df1.select_dtypes(include = "object").columns
object_columns


# In[52]:


import pickle


# In[85]:


# initialize a dictinory to save the encoders
encoders = {}

# apply label encoding and store the encoders
for column in object_columns:
    label_encoder = LabelEncoder()
    df1[column] = label_encoder.fit_transform(df1[column])
    encoders[column] = label_encoder
    
# save the encoders to a pickle file
with open("encoders.pkl","wb") as f:
    pickle.dump(encoders, f)
    
with open('model.pkl', 'wb') as f:
    pickle.dump({"model": df1, "encoders": encoders}, f)


# In[54]:


encoders


# In[55]:


df1.head(10)


# ### Training and Data split

# In[56]:


#splting  the features and target
X = df1.drop(columns = ["Churn"])
y = df1["Churn"]


# In[57]:


y


# In[58]:


from sklearn.model_selection import train_test_split,cross_val_score


# In[59]:


# split training and test data
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,
                                                 random_state=42)


# In[60]:


y_train.shape


# In[61]:


y_train.value_counts()


# ### synthesic minority oversampling technique(SMOTE)

# In[62]:


from imblearn.over_sampling import SMOTE


# In[63]:


smote = SMOTE(random_state=42)


# In[64]:


X_train_smote, y_train_smote = smote.fit_resample(X_train,y_train)


# In[65]:


y_train_smote.shape


# In[66]:


y_train_smote.value_counts()


# # Model Training

# In[67]:


### training with default parameters


# In[68]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# In[69]:


# dictionary of models
models = {
    "Decision Tree":DecisionTreeClassifier(random_state = 42),
    "Random forest":RandomForestClassifier(random_state = 42),
    "XGBoost":XGBClassifier(random_state = 42),
}


# In[70]:


# dictinoary to store the cross validation results
cv_scores = {}

# perform fold cross validation for each model
for model_name, model in models.items():
    print(f"Training {model_name} with default parameters")
    scores = cross_val_score(model,X_train_smote,y_train_smote,
                             cv=5,scoring = "accuracy")
    cv_scores[model_name] = scores
    print(f"{model_name} cross_validations accuracy : {np.mean(scores):.2f}")
    print("-"*70)


# In[71]:


cv_scores


# - Random Forest gives the heighest accuracy comapred to other models with default parameters

# In[72]:


rfc = RandomForestClassifier(random_state=42)
#from sklearn.ensemble import RandomForestClassifier
#import pickle

# Train your model
#rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
#rf_model.fit(X_train, y_train)  

# Save the trained model
#with open("random_forest_model.pkl", "wb") as f:
 #   pickle.dump(rf_model, f)


# In[73]:


rfc.fit(X_train,y_train)


# In[74]:


y_test.value_counts()


# # Model Evaluation

# In[75]:


from sklearn.metrics import accuracy_score,confusion_matrix, classification_report


# In[76]:


# evaluate on test data
y_test_pred = rfc.predict(X_test)

print("Accuracy Score: \n",accuracy_score(y_test,y_test_pred))
print('Confusion matrix: \n',confusion_matrix(y_test,y_test_pred))
print('Classification Report: \n',classification_report(y_test,y_test_pred))


# In[77]:


# save the trained model as a pickle file
model_data = {"model": rfc, "features_names": X.columns.tolist()}

with open("customer_churn_model.pkl", "wb") as f:
    pickle.dump(model_data, f)


# # Load the same model and build a predictive system
# 

# In[78]:


# load the saved model and the features names
with open("customer_churn_model.pkl","rb") as f:
    model_data = pickle.load(f)
    
loaded_model = model_data["model"]
fearure_names = model_data["features_names"]


# In[79]:


loaded_model


# In[80]:


fearure_names


# In[94]:


import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Sample Data (Replace with actual dataset)
data = pd.DataFrame({
    "gender": ["Female", "Male", "Female", "Male"],
    "SeniorCitizen": [0, 1, 0, 1],
    "Partner": ["Yes", "No", "Yes", "No"],
    "Dependents": ["No", "No", "Yes", "No"],
    "tenure": [1, 34, 5, 20],
    "PhoneService": ["No", "Yes", "Yes", "Yes"],
    "MultipleLines": ["No phone service", "Yes", "No", "No"],
    "InternetService": ["DSL", "Fiber optic", "DSL", "Fiber optic"],
    "OnlineSecurity": ["No", "Yes", "No", "Yes"],
    "OnlineBackup": ["Yes", "No", "Yes", "No"],
    "DeviceProtection": ["No", "Yes", "No", "Yes"],
    "TechSupport": ["No", "Yes", "No", "No"],
    "StreamingTV": ["No", "Yes", "No", "Yes"],
    "StreamingMovies": ["No", "Yes", "No", "Yes"],
    "Contract": ["Month-to-month", "Two year", "One year", "Month-to-month"],
    "PaperlessBilling": ["Yes", "No", "Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Electronic check", "Credit card"],
    "MonthlyCharges": [29.85, 56.95, 42.30, 78.85],
    "TotalCharges": [29.85, 1889.50, 204.45, 2203.45],
    "Churn": ["No", "Yes", "No", "Yes"]  # Target Variable
})

# Separate Features & Target
X = data.drop(columns=["Churn"])
y = data["Churn"]

# Encode Target Variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # "No" -> 0, "Yes" -> 1

# Encode Categorical Features
encoders = {}  # Dictionary to store encoders for later use
for column in X.select_dtypes(include=["object"]).columns:
    encoder = LabelEncoder()
    X[column] = encoder.fit_transform(X[column])
    encoders[column] = encoder  # Save encoder

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save Model & Encoders
with open('model.pkl', 'wb') as f:
    pickle.dump({"model": model, "encoders": encoders, "label_encoder": label_encoder}, f)


# In[96]:


# Load Model & Encoders
with open('model.pkl', 'rb') as f:
    saved_data = pickle.load(f)
    model = saved_data["model"]
    encoders = saved_data["encoders"]
    label_encoder = saved_data["label_encoder"]

# New Data for Prediction
input_data = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}

# Convert to DataFrame
input_data_df = pd.DataFrame([input_data])

# Encode Categorical Features
for column, encoder in encoders.items():
    if column in input_data_df.columns:
        input_data_df[column] = encoder.transform(input_data_df[[column]])

# Make Predictions
prediction = model.predict(input_data_df)
pred_prob = model.predict_proba(input_data_df)

# Decode Prediction
predicted_class = label_encoder.inverse_transform([prediction[0]])[0]

# Output Results
print(f"Prediction: {predicted_class}")  # "No" or "Yes"
print(f"Prediction Probability: {pred_prob}")


# In[91]:


import sklearn
print(sklearn.__version__)



# In[ ]:




