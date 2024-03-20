#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


# In[2]:


Stroke_dataset = pd.read_csv('C:/Users/shali/Documents/Research project/healthcare-dataset-stroke-data.csv')


# In[3]:


Stroke_dataset.head()


# In[4]:


Stroke_dataset.describe()


# In[5]:


print(Stroke_dataset.dtypes)


# In[6]:


Stroke_dataset['stroke'].value_counts()


# In[7]:


Stroke_dataset.isnull().sum()


# In[8]:


Stroke_dataset.dropna()


# In[9]:


#  import pandas as pd
# import numpy as np
# from tkinter import filedialog
# from tkinter import Tk
# from sklearn.preprocessing import LabelEncoder

# # Instantiate Tkinter window
# root = Tk()
# root.withdraw()  # Hide the main window

# # Open dialog to select file
# file_path = filedialog.askopenfilename()

# # Load the dataset
# data = pd.read_csv(file_path)

# # Identify numerical and categorical columns
# num_cols = data.select_dtypes(include=np.number).columns.tolist()
# cat_cols = data.select_dtypes(include='object').columns.tolist()

# # Print numerical and categorical columns
# print("Numerical columns: ", num_cols)
# print("Categorical columns: ", cat_cols)

# # Check if the dataset is balanced or not
# target_col = 'Class'  # Replace this with your target column

# # Convert the target column to numerical if it is categorical
# if data[target_col].dtype == 'object':
#     data[target_col] = LabelEncoder().fit_transform(data[target_col])

# counts = data[target_col].value_counts()
# print("\nCounts of each class in the target column:")
# print(counts)

# min_count = np.min(counts)
# max_count = np.max(counts)

# if max_count / min_count > 1.5:  # This threshold can be adjusted based on your requirements
#     print("\nThe dataset is imbalanced.")
# else:
#     print("\nThe dataset is balanced.")


# In[10]:


#Checking unique values for columns 
columns_to_check = ['gender', 'ever_married', 'work_type','Residence_type','smoking_status']  # add column names

for column in columns_to_check:
    print(f"Unique values in {column}: {Stroke_dataset[column].unique()}")


# In[11]:


# import seaborn as sns
print(Stroke_dataset['gender'].value_counts())
sns.countplot(x='gender', data=Stroke_dataset)
plt.title('Gender graph')
plt.show()


# In[ ]:





# In[12]:


#Transfering character data to numerical data for the above columns


from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Initialize one-hot encoder
encoder = OneHotEncoder()

# Assume 'dataset' is your DataFrame
columns_to_encode = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

# Fit and transform data with encoder
data_encoded = encoder.fit_transform(Stroke_dataset[columns_to_encode])

# Create a DataFrame from the sparse matrix
data_encoded_df = pd.DataFrame(data_encoded.toarray(), columns=encoder.get_feature_names(columns_to_encode))

# Drop original columns from dataset
Stroke_dataset = Stroke_dataset.drop(columns_to_encode, axis=1)

# Concatenate original dataset with the encoded DataFrame
Stroke_dataset = pd.concat([Stroke_dataset, data_encoded_df], axis=1)


# In[13]:


Stroke_dataset.head()


# In[14]:


Stroke_dataset.corr()['stroke'].sort_values(ascending=False) 


# In[15]:


# Plotting the correlation graph

correlation_matrix = Stroke_dataset.corr()

plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".1f", square=True, cmap = 'cool')
plt.show()


# In[ ]:





# In[16]:


# import seaborn as sns
sns.countplot(x='stroke', data=Stroke_dataset)
plt.title('0: No       1: Yes')
plt.show()


# In[ ]:





# In[17]:


# Above graph shows the dataset is imbalanced
# Performing preprocessing techniques to handle imbalanced dataset 


# In[18]:


np.any(np.isnan(Stroke_dataset))


# In[19]:


Stroke_dataset.isnull().sum()


# In[20]:


Stroke_dataset.drop(['id'], axis=1)


# In[21]:


Stroke_dataset  = Stroke_dataset.dropna()
Stroke_dataset.tail()


# In[22]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

Stroke_dataset = pd.DataFrame(scaler.fit_transform(Stroke_dataset), columns=Stroke_dataset.columns)


# In[23]:


# Splitting the dataset into two more datasets for Male only and female only dataset
Stroke_dataset_male  = Stroke_dataset[Stroke_dataset['gender_Male'] == 1]
Stroke_dataset_male.drop(['gender_Female'], axis=1)


# In[24]:


Stroke_dataset_female  = Stroke_dataset[Stroke_dataset['gender_Female'] == 1.0]
Stroke_dataset_female.drop(['gender_Male'], axis=1)


# In[25]:


X = Stroke_dataset.drop(['stroke'], axis=1)
y = Stroke_dataset['stroke']


# In[26]:


# While sampling before splititng may allow data leakage as we  need the test data to be completely unknown 


# In[27]:


from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[28]:


print((X_train['gender_Male'].value_counts()))
print ((X_train['gender_Female'].value_counts())) 


# In[29]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

def logistic_regression_with_metrics(X_train, y_train, X_test, y_test):
    # Initializing the Logistic Regression model
    model = LogisticRegression()

    # Fitting the model on the training data and making predictions
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy*100:.2f}%')

    # Computing confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    print('Confusion matrix:')
    print(conf_mat)

    # Plotting confusion matrix
    plot_confusion_matrix(model, X_test, y_test)  
    plt.show()  

    return y_pred, accuracy


# In[30]:


from xgboost import XGBClassifier


def xgboost_with_metrics(X_train, y_train, X_test, y_test):
    # Initializing the XGBoost classifier
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # Fitting the model on the training data and making predictions
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy*100:.2f}%')

    # Computing confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    print('Confusion matrix:')
    print(conf_mat)

    # Plotting confusion matrix
    plot_confusion_matrix(model, X_test, y_test)  
    plt.show()  

    return y_pred, accuracy


# In[31]:


from sklearn.ensemble import RandomForestClassifier
def random_forest_with_metrics(X_train, y_train, X_test, y_test):
    # Initializing the Random Forest classifier
    model = RandomForestClassifier()

    # Fitting the model on the training data and making predictions
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy*100:.2f}%')

    # Computing confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    print('Confusion matrix:')
    print(conf_mat)

    # Plotting confusion matrix
    plot_confusion_matrix(model, X_test, y_test)  
    plt.show()  

    return y_pred, accuracy


# In[32]:


# results before sampling techniques

y_pred, accuracy = logistic_regression_with_metrics(X_train, y_train, X_test, y_test)


# In[33]:


y_pred, accuracy = xgboost_with_metrics(X_train, y_train, X_test, y_test)


# In[34]:



y_pred, accuracy = random_forest_with_metrics(X_train, y_train, X_test, y_test)


# In[35]:



sm = SMOTE(random_state=42)

X_train, y_train = sm.fit_resample(X_train, y_train)

#Class distribution after SMOTE
print("Resampled class distribution :", Counter(y_train))
print("Resampled gender distribution :", X_train['gender_Male'].value_counts())


# In[36]:


print((X_train['gender_Male'].value_counts()))
print ((X_train['gender_Female'].value_counts()))


# In[37]:


# np.all(np.isfinite(dataset))


# In[38]:


# dataset.isnull().sum()


# In[39]:


# Prediction after sampling
y_pred, accuracy = logistic_regression_with_metrics(X_train, y_train, X_test, y_test)


# In[40]:


y_pred, accuracy = xgboost_with_metrics(X_train, y_train, X_test, y_test)


# In[41]:



y_pred, accuracy = random_forest_with_metrics(X_train, y_train, X_test, y_test)


# In[42]:


# Training using male only dataset


# In[43]:


X = Stroke_dataset_male.drop(['stroke'], axis=1)
y = Stroke_dataset_male['stroke']

# X
# y


# In[44]:


# Splitting 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[45]:


# Male only dataset

y_pred, accuracy = logistic_regression_with_metrics(X_train, y_train, X_test, y_test)


# In[46]:


y_pred, accuracy = xgboost_with_metrics(X_train, y_train, X_test, y_test)


# In[47]:



y_pred, accuracy = random_forest_with_metrics(X_train, y_train, X_test, y_test)


# In[48]:


# Using male only train dataset to predict for female only dataset


# In[49]:


X_train = Stroke_dataset_male.drop(['stroke'], axis=1)
y_train = Stroke_dataset_male['stroke']
X_test = Stroke_dataset_female.drop(['stroke'],axis = 1)
y_test = Stroke_dataset_female['stroke']
#X_train
# y_train


# In[50]:


# # Splitting and Resampling

# X_train, y_train = sm.fit_resample(X_train, y_train)


# In[51]:


# Male only dataset

y_pred, accuracy = logistic_regression_with_metrics(X_train, y_train, X_test, y_test)


# In[52]:


y_pred, accuracy = xgboost_with_metrics(X_train, y_train, X_test, y_test)


# In[53]:



y_pred, accuracy = random_forest_with_metrics(X_train, y_train, X_test, y_test)


# In[54]:


# Training female only dataset for training,testing
# Splitting and Resampling
X = Stroke_dataset_female.drop(['stroke'],axis=1)
y = Stroke_dataset_female['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:





# In[55]:


y_pred, accuracy = logistic_regression_with_metrics(X_train, y_train, X_test, y_test)


# In[56]:


y_pred, accuracy = xgboost_with_metrics(X_train, y_train, X_test, y_test)


# In[57]:



y_pred, accuracy = random_forest_with_metrics(X_train, y_train, X_test, y_test)


# In[58]:


# Using female dataset on male data prediction
X_train = Stroke_dataset_female.drop(['stroke'], axis=1)
y_train = Stroke_dataset_female['stroke']
X_test = Stroke_dataset_male.drop(['stroke'],axis = 1)
y_test = Stroke_dataset_male['stroke']
#X_train
# y_train


# In[59]:


# Resampling and predicting

X_train, y_train = sm.fit_resample(X_train, y_train)
y_pred, accuracy = logistic_regression_with_metrics(X_train, y_train, X_test, y_test)


# In[60]:


y_pred, accuracy = xgboost_with_metrics(X_train, y_train, X_test, y_test)


# In[61]:



y_pred, accuracy = random_forest_with_metrics(X_train, y_train, X_test, y_test)


# In[62]:



#     Heart_dataset = pd.read_csv('C:/Users/shali/Documents/Research project/heart_2020_cleaned.csv')
        #     Heart_dataset.head()


# In[63]:


# Heart_dataset.rename(columns=Heart_dataset.iloc[0], inplace = True)
# Heart_dataset.head()


# In[64]:


# Heart_dataset=Heart_dataset.drop(index=0)


# In[65]:


# # import seaborn as sns
# print(Heart_dataset['Sex'].value_counts())
# sns.countplot(x='Sex', data=Heart_dataset)
# plt.title('Gender graph')
# plt.show()
# # import seaborn as sns
# print(Heart_dataset['HeartDisease'].value_counts())
# sns.countplot(x='HeartDisease', data=Heart_dataset)
# plt.title('Heart Disease graph')
# plt.show()


# In[66]:





# Heart_dataset.isnull().sum()


# In[67]:






# #Checking unique values for columns 
# columns_to_check = ['HeartDisease', 'GenHealth','Smoking', 'AlcoholDrinking','DiffWalking','Sex','Race','Diabetic','PhysicalActivity','Asthma','KidneyDisease','SkinCancer']  # add column names

# for column in columns_to_check:
#     print(f"Unique values in {column}: {Heart_dataset[column].unique()}")


# In[68]:


# columns_to_encode = ['HeartDisease', 'Smoking','GenHealth', 'AlcoholDrinking','DiffWalking','Sex','Race','Diabetic','PhysicalActivity','Asthma','KidneyDisease','SkinCancer'] 

# # Fit and transform data with encoder
# data_encoded = encoder.fit_transform(Heart_dataset[columns_to_encode])

# # Create a DataFrame from the sparse matrix
# data_encoded_df = pd.DataFrame(data_encoded.toarray(), columns=encoder.get_feature_names(columns_to_encode))

# # Drop original columns from dataset
# Heart_dataset = Heart_dataset.drop(columns_to_encode, axis=1)

# # Concatenate original dataset with the encoded DataFrame
# Heart_dataset = pd.concat([Heart_dataset, data_encoded_df], axis=1)


# In[69]:


# Heart_dataset.dropna()


# In[70]:


#  Heart_dataset['AgeCategory'] = Heart_dataset['AgeCategory'].str[:2]
# Heart_dataset['AgeCategory']


# In[71]:


# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()

# Heart_dataset['Stroke'] = le.fit_transform(Heart_dataset['Stroke'])


# In[72]:


# Heart_dataset.isnull().sum()


# In[73]:


# Heart_dataset=Heart_dataset.dropna()


# In[74]:


# X = Heart_dataset.drop(['Stroke'], axis=1)

# y = Heart_dataset['Stroke']


# In[75]:


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[76]:




# y_pred, accuracy = logistic_regression_with_metrics(X_train, y_train, X_test, y_test)


# In[77]:


# # Splitting the dataset into two more datasets for Male only and female only dataset
# Heart_dataset_male  = Heart_dataset[Heart_dataset['Sex_Male'] == 1.0]
# Heart_dataset_male.drop(['Sex_Female'], axis=1)


# In[78]:


# Heart_dataset_female = Heart_dataset[Heart_dataset['Sex_Female'] == 1.0]
# Heart_dataset_female.drop(['Sex_Male'],axis = 1)


# In[79]:


# X = Heart_dataset_male.drop(['Stroke'], axis=1)
# y = Heart_dataset_male['Stroke']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# y_pred, accuracy = logistic_regression_with_metrics(X_train, y_train, X_test, y_test)


# In[80]:


# X_train = Heart_dataset_male.drop(['Stroke'], axis=1)
# y_train= Heart_dataset_male['Stroke']
# X_test = Heart_dataset_female.drop(['Stroke'],axis = 1)
# y_test = Heart_dataset_female['Stroke']
# y_pred, accuracy = logistic_regression_with_metrics(X_train, y_train, X_test, y_test)


# In[81]:


# X = Heart_dataset_female.drop(['Stroke'], axis=1)
# y = Heart_dataset_female['Stroke']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# y_pred, accuracy = logistic_regression_with_metrics(X_train, y_train, X_test, y_test)


# In[82]:


# X_train = Heart_dataset_female.drop(['Stroke'], axis=1)
# y_train= Heart_dataset_female['Stroke']
# X_test = Heart_dataset_male.drop(['Stroke'],axis = 1)
# y_test = Heart_dataset_male['Stroke']
# y_pred, accuracy = logistic_regression_with_metrics(X_train, y_train, X_test, y_test)


# In[83]:


# # Using Smote for ReSampling and using it on 1st dataset
# X = Stroke_dataset.drop(['stroke'],axis = 1)
# y = Stroke_dataset['stroke']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# sm = SMOTE(random_state=42)

# X_train, y_train = sm.fit_resample(X_train, y_train)

# #Class distribution after SMOTE
# print("Resampled class distribution :", Counter(y_train))
# print("Resampled gender distribution :", X_train['gender_Male'].value_counts())


# In[84]:


# y_pred, accuracy = logistic_regression_with_metrics(X_train, y_train, X_test, y_test)


# In[85]:


#!pip install aif360


# In[86]:


# !pip install tensorflow


# In[87]:



# # Libraries to study
# from aif360.datasets import StandardDataset
# from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
# from aif360.algorithms.preprocessing import LFR, Reweighing
# from aif360.algorithms.inprocessing import AdversarialDebiasing, PrejudiceRemover
# from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing, EqOddsPostprocessing, RejectOptionClassification

# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()

# # ML libraries
# from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve, auc
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# import tensorflow as tf

# # Design libraries
# from IPython.display import Markdown, display
# import warnings
# warnings.filterwarnings("ignore")

# # Converting to aif360 dataset format
# dataset = StandardDataset(Stroke_dataset, label_name='stroke', 
#                           protected_attribute_names=['gender_Female', 'gender_Male', 'gender_Other'],
#                           privileged_classes=[[1.0], [0.0], [0.0]], 
#                           favorable_classes=[1])

# # Split data into training and test
# train, test = dataset.split([0.7], shuffle=True)

# # Train Adversarial Debiasing model
# sess = tf.compat.v1.Session()
# adv_debias = AdversarialDebiasing(privileged_groups=[{'gender_Female': 1.0}],
#                                   unprivileged_groups=[{'gender_Female': 0.0}],
#                                   scope_name="adv_debias",
#                                   debias=True,
#                                   sess=sess)

# adv_debias.fit(train)

# # Predictions
# y_train_pred = adv_debias.predict(train)
# y_test_pred = adv_debias.predict(test)

# # Evaluate fairness
# metric_train = BinaryLabelDatasetMetric(y_train_pred, 
#                                         unprivileged_groups=[{'gender_Female': 0.0}],
#                                         privileged_groups=[{'gender_Female': 1.0}])
# metric_test = BinaryLabelDatasetMetric(y_test_pred, 
#                                        unprivileged_groups=[{'gender_Female': 0.0}],
#                                        privileged_groups=[{'gender_Female': 1.0}])

# print("Train dataset difference in mean outcomes between unprivileged and privileged groups:", metric_train.mean_difference())
# print("Test dataset difference in mean outcomes between unprivileged and privileged groups:", metric_test.mean_difference())

# # Close the TF session
# sess.close()


# In[88]:


# from aif360.datasets import BinaryLabelDataset
# from aif360.algorithms.inprocessing import AdversarialDebiasing
# from aif360.metrics import BinaryLabelDatasetMetric
# from sklearn.metrics import accuracy_score
# import tensorflow as tf

# # Assuming you have Stroke_dataset defined
# X = Stroke_dataset.drop('stroke', axis=1)
# y = Stroke_dataset['stroke']

# # Splitting for train-test
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import f1_score
# from imblearn.over_sampling import ADASYN

# # Provided train-test split:
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# # Use ADASYN for resampling
# adasyn = ADASYN(random_state=42)
# X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)

# # Train logistic regression on the ADASYN resampled data
# logistic_model = LogisticRegression(max_iter=10000)
# logistic_model.fit(X_train_resampled, y_train_resampled)

# # Make predictions using logistic regression on the test data
# logistic_preds = logistic_model.predict(X_test)

# # Calculate F1 score
# f1_logistic = f1_score(y_test, logistic_preds)

# print("F1 Score for Logistic Regression on ADASYN resampled data:", f1_logistic)


# In[89]:


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import ADASYN




# tf.compat.v1.disable_eager_execution()

# privileged_groups = [{'gender_Female': 1}]
# unprivileged_groups = [{'gender_Female': 0}]

# # Convert the dataset into BinaryLabelDataset format for aif360
# train_bld = BinaryLabelDataset(df=pd.concat((X_train_resampled, y_train_resampled), axis=1),
#                                label_names=['stroke'], protected_attribute_names=['gender_Male', 'gender_Female'])

# # Initialize the Adversarial Debiasing model
# sess = tf.compat.v1.Session()
# adv_debiasing = AdversarialDebiasing(privileged_groups=privileged_groups,
#                                      unprivileged_groups=unprivileged_groups,
#                                      scope_name="adv_debiasing", debias=True, sess=sess)

# # Train the model
# adv_debiasing.fit(train_bld)

# # Obtain predictions on train data
# train_preds = adv_debiasing.predict(train_bld)

# # Continue with your evaluation using confusion matrix or other metrics


# In[ ]:





# In[90]:


# # Convert the resampled test data into BinaryLabelDataset format
# test_bld_resampled = BinaryLabelDataset(df=pd.concat((X_test, y_test), axis=1),
#                                         label_names=['stroke'], protected_attribute_names=['gender_Male', 'gender_Female'])

# # Make predictions on resampled dataa
# y_pred_adv_resampled = adv_debiasing.predict(test_bld_resampled).labels.flatten()


# In[91]:


# from sklearn.metrics import confusion_matrix

# cm_adv_resampled = confusion_matrix(y_test, y_pred_adv_resampled)
# cm_adv_resampled


# In[92]:


# import matplotlib.pyplot as plt
# import seaborn as sns

# def plot_confusion_matrix(cm, title):
#     plt.figure(figsize=(6, 4))
#     sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', cbar=False, 
#                 xticklabels=['Predicted 0', 'Predicted 1'], 
#                 yticklabels=['Actual 0', 'Actual 1'])
#     plt.title(title)
#     plt.show()

# plot_confusion_matrix(cm_adv_resampled, "Confusion Matrix - Adversarial Debiasing on Resampled Data")


# In[93]:


# # 3. Calculate and compare F1 scores
# from sklearn.metrics import f1_score
# f1_logistic = f1_score(y_test, logistic_preds)
# f1_adv_debiasing = f1_score(y_test, y_pred_adv_resampled)

# print("F1 Score for Logistic Regression:", f1_logistic)
# print("F1 Score for Adversarial Debiasing:", f1_adv_debiasing)


# In[94]:


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from aif360.datasets import BinaryLabelDataset
# from aif360.algorithms.inprocessing import AdversarialDebiasing

# tf.compat.v1.disable_eager_execution()

# privileged_groups = [{'gender_Female': 1}]
# unprivileged_groups = [{'gender_Female': 0}]

# # Assuming you've loaded your original dataset into X and y
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Convert the dataset into BinaryLabelDataset format for aif360
# train_bld = BinaryLabelDataset(df=pd.concat((X_train, y_train), axis=1),
#                                label_names=['stroke'], protected_attribute_names=['gender_Male', 'gender_Female'])

# # Initialize the Adversarial Debiasing model
# sess = tf.compat.v1.Session()
# adv_debiasing = AdversarialDebiasing(privileged_groups=privileged_groups,
#                                      unprivileged_groups=unprivileged_groups,
#                                      scope_name="adv_debiasing", debias=True, sess=sess)

# # Train the model
# adv_debiasing.fit(train_bld)

# # Obtain predictions on train data
# train_preds = adv_debiasing.predict(train_bld)

# # Continue with your evaluation using confusion matrix or other metrics


# In[95]:


# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import itertools

# # Extract true and predicted labels from BinaryLabelDataset format
# true_labels = train_bld.labels.ravel()
# predicted_labels = train_preds.labels.ravel()

# # Compute the confusion matrix using sklearn
# cm = confusion_matrix(true_labels, predicted_labels)

# # Plotting the confusion matrix
# plt.figure(figsize=(10,7))
# plt.title('Confusion Matrix')
# plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# plt.colorbar()

# # Add labels to the plot
# classes = [0, 1]  # Assuming binary classification
# tick_marks = np.arange(len(classes))
# plt.xticks(tick_marks, classes)
# plt.yticks(tick_marks, classes)

# # Display values inside the matrix
# thresh = cm.max() / 2.
# for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#     plt.text(j, i, format(cm[i, j], 'd'),
#              horizontalalignment="center",
#              color="white" if cm[i, j] > thresh else "black")

# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.tight_layout()
# plt.show()


# In[96]:


# # Convert the test dataset into BinaryLabelDataset format for aif360
# test_bld = BinaryLabelDataset(df=pd.concat((X_test, y_test), axis=1),
#                               label_names=['stroke'], protected_attribute_names=['gender_Male', 'gender_Female'])

# # Obtain predictions on test data
# test_preds = adv_debiasing.predict(test_bld)

# # Extract true and predicted labels from BinaryLabelDataset format for test data
# true_test_labels = test_bld.labels.ravel()
# predicted_test_labels = test_preds.labels.ravel()

# # Compute the confusion matrix using sklearn for test data
# cm_test = confusion_matrix(true_test_labels, predicted_test_labels)

# # Plotting the confusion matrix for test data
# plt.figure(figsize=(10,7))
# plt.title('Test Confusion Matrix')
# plt.imshow(cm_test, interpolation='nearest', cmap=plt.cm.Blues)
# plt.colorbar()

# # Add labels to the plot
# classes = [0, 1]  # Assuming binary classification
# tick_marks = np.arange(len(classes))
# plt.xticks(tick_marks, classes)
# plt.yticks(tick_marks, classes)

# # Display values inside the matrix
# thresh = cm_test.max() / 2.
# for i, j in itertools.product(range(cm_test.shape[0]), range(cm_test.shape[1])):
#     plt.text(j, i, format(cm_test[i, j], 'd'),
#              horizontalalignment="center",
#              color="white" if cm_test[i, j] > thresh else "black")

# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.tight_layout()
# plt.show()


# In[97]:


# from sklearn.metrics import accuracy_score

# # Using the true_test_labels and predicted_test_labels obtained earlier
# accuracy = accuracy_score(true_test_labels, predicted_test_labels)
# print("Accuracy Score:", accuracy)


# In[98]:


X_train = Stroke_dataset_male.drop(['stroke'], axis=1)
y_train = Stroke_dataset_male['stroke']
X_test = Stroke_dataset_female.drop(['stroke'],axis = 1)
y_test = Stroke_dataset_female['stroke']
#X_train
y_train
# X = Stroke_dataset.drop(['stroke'], axis=1)
# y = Stroke_dataset['stroke']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train, y_train= smote.fit_resample(X_train, y_train)

print("After Oversampling with SMOTE:")
print(y_train.value_counts())


# In[99]:


y_pred_logistic, accuracy = logistic_regression_with_metrics(X_train, y_train, X_test, y_test)


# In[100]:



import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Assuming the function returns the predicted probabilities for the positive class
y_pred_xgboost, accuracy = xgboost_with_metrics(X_train, y_train, X_test, y_test)

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_xgboost)
roc_auc = auc(fpr, tpr)

# Plotting
plt.figure(figsize=(10, 7))
lw = 2

plt.plot(fpr, tpr, color='blue', lw=lw, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')  # Diagonal reference line

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[101]:





import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Assuming the function returns the predicted probabilities for the positive class
y_pred_random, accuracy = random_forest_with_metrics(X_train, y_train, X_test, y_test)

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_random)
roc_auc = auc(fpr, tpr)

# Plotting
plt.figure(figsize=(10, 7))
lw = 2

plt.plot(fpr, tpr, color='blue', lw=lw, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')  # Diagonal reference line

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[102]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Assuming the function returns the predicted probabilities for the positive class
y_pred_logistic, accuracy = logistic_regression_with_metrics(X_train, y_train, X_test, y_test)

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_logistic)
roc_auc = auc(fpr, tpr)

# Plotting
plt.figure(figsize=(10, 7))
lw = 2

plt.plot(fpr, tpr, color='blue', lw=lw, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')  # Diagonal reference line

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[103]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.inprocessing import AdversarialDebiasing

tf.compat.v1.disable_eager_execution()

privileged_groups = [{'gender_Female': 1}]
unprivileged_groups = [{'gender_Male': 1}]


# Converting the dataset into BinaryLabelDataset format for aif360
train_bld = BinaryLabelDataset(df=pd.concat((X_train, y_train), axis=1),
                               label_names=['stroke'], protected_attribute_names=['gender_Male', 'gender_Female'])

# Initializing the Adversarial Debiasing model
sess = tf.compat.v1.Session()
adv_debiasing = AdversarialDebiasing(privileged_groups=privileged_groups,
                                     unprivileged_groups=unprivileged_groups,
                                     scope_name="adv_debiasing", debias=True, sess=sess)

# Train the model
adv_debiasing.fit(train_bld)

# Obtain predictions on train data
train_preds = adv_debiasing.predict(train_bld)

# Continue with your evaluation using confusion matrix or other metrics


# In[104]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

# Extract true and predicted labels from BinaryLabelDataset format
true_labels = train_bld.labels.ravel()
predicted_labels = train_preds.labels.ravel()

# Compute the confusion matrix using sklearn
cm = confusion_matrix(true_labels, predicted_labels)

# Plotting the confusion matrix
plt.figure(figsize=(10,7))
plt.title('Confusion Matrix')
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()

# Add labels to the plot
classes = [0, 1]  # Assuming binary classification
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

# Display values inside the matrix
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()


# In[105]:


from sklearn.metrics import accuracy_score

# Using the true_test_labels and predicted_test_labels obtained earlier
accuracy = accuracy_score(true_labels, predicted_labels)
print("Accuracy Score:", accuracy)


# In[106]:


# Convert the test dataset into BinaryLabelDataset format for aif360
test_bld = BinaryLabelDataset(df=pd.concat((X_test, y_test), axis=1),
                              label_names=['stroke'], protected_attribute_names=['gender_Male', 'gender_Female'])

# Obtain predictions on test data
test_preds = adv_debiasing.predict(test_bld)

# Extract true and predicted labels from BinaryLabelDataset format for test data
true_test_labels = test_bld.labels.ravel()
predicted_test_labels = test_preds.labels.ravel()

# Compute the confusion matrix using sklearn for test data
cm_test = confusion_matrix(true_test_labels, predicted_test_labels)

# Plotting the confusion matrix for test data
plt.figure(figsize=(10,7))
plt.title('Test Confusion Matrix')
plt.imshow(cm_test, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()

# Add labels to the plot
classes = [0, 1]  # Assuming binary classification
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

# Display values inside the matrix
thresh = cm_test.max() / 2.
for i, j in itertools.product(range(cm_test.shape[0]), range(cm_test.shape[1])):
    plt.text(j, i, format(cm_test[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm_test[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()


# In[107]:


from sklearn.metrics import accuracy_score

# Using the true_test_labels and predicted_test_labels obtained earlier
accuracy = accuracy_score(true_test_labels, predicted_test_labels)
print("Accuracy Score:", accuracy)


# In[108]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Compute ROC curve and AUC for Logistic Regression
fpr_logistic, tpr_logistic, _ = roc_curve(y_test, y_pred_logistic)
roc_auc_logistic = auc(fpr_logistic, tpr_logistic)


fpr_xgboost, tpr_xgboost, _ = roc_curve(y_test, y_pred_xgboost)
roc_auc_xgboost = auc(fpr_xgboost, tpr_xgboost)

fpr_random, tpr_random, _ = roc_curve(y_test, y_pred_random)
roc_auc_random = auc(fpr_random, tpr_random)


# Compute ROC curve and AUC for Adversarial Debiasing
fpr_adv_debias, tpr_adv_debias, _ = roc_curve(true_test_labels, predicted_test_labels)
roc_auc_adv_debias = auc(fpr_adv_debias, tpr_adv_debias)

# Plotting
plt.figure(figsize=(10, 7))
lw = 2

# Plotting ROC curve for Logistic Regression
plt.plot(fpr_logistic, tpr_logistic, color='red', lw=lw, label=f'Logistic Regression (AUC = {roc_auc_logistic:.2f})')
plt.plot(fpr_xgboost, tpr_xgboost, color='blue', lw=lw, label=f'XgBoost (AUC = {roc_auc_xgboost:.2f})')
plt.plot(fpr_random, tpr_random, color='yellow', lw=lw, label=f'Random Forest (AUC = {roc_auc_random:.2f})')

# Plotting ROC curve for Adversarial Debiasing
plt.plot(fpr_adv_debias, tpr_adv_debias, color='green', lw=lw, label=f'Adversarial Debiasing (AUC = {roc_auc_adv_debias:.2f})')

plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[109]:


Stroke_dataset = Stroke_dataset.drop(['gender_Female', 'gender_Male', 'gender_Other'], axis=1)


# In[110]:


X = Stroke_dataset.drop(['stroke'], axis=1)
y = Stroke_dataset['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train, y_train= smote.fit_resample(X_train, y_train)

print("After Oversampling with SMOTE:")
print(y_train.value_counts())


# In[111]:


y_pred_logistic, accuracy = logistic_regression_with_metrics(X_train, y_train, X_test, y_test)


# In[112]:


y_pred_random, accuracy = random_forest_with_metrics(X_train, y_train, X_test, y_test)


# In[113]:


y_pred_xgboost, accuracy = xgboost_with_metrics(X_train, y_train, X_test, y_test)


# In[114]:


import matplotlib.pyplot as plt

# Model names
models = ['Logistic Regression', 'Random Forest', 'XgBoost']

# Accuracies with gender column
accuracies_with_gender = [73.39,  92.80, 92.74]

# Accuracies without gender column
accuracies_without_gender = [72.78,  91.17, 93.21]

# Set up the figure and axis
plt.figure(figsize=(10, 6))
plt.plot(models, accuracies_with_gender, marker='o', label='With Gender Column')
plt.plot(models, accuracies_without_gender, marker='o', linestyle='--', label='Without Gender Column')

# Set title and labels
plt.title('Model Accuracies With and Without Gender Column')
plt.ylabel('Accuracy (%)')
plt.xlabel('Models')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()

# Show plot
plt.show()


# In[ ]:




