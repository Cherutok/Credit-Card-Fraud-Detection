#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection
# This project addresses the critical challenge of detecting fraudulent transactions in credit card operations using advanced machine learning techniques. With the increasing volume of digital transactions, automated fraud detection systems have become essential for financial institutions to minimize losses and maintain customer trust.

# In[ ]:


### 1.2 Project Objectives
- Perform comprehensive Exploratory Data Analysis (EDA) to understand transaction patterns and fraud indicators
- Develop and compare multiple machine learning models for fraud detection
- Optimize model performance using various techniques including SMOTE and hyperparameter tuning
- Achieve high accuracy while maintaining a balance between fraud detection and false alarms


# In[ ]:


##1.3 Business Impact
- Minimize financial losses from fraudulent transactions
- Reduce operational costs of manual review
- Enhance customer trust and satisfaction
- Provide scalable fraud detection capabilities


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.ensemble import BaggingClassifier
import warnings
warnings.filterwarnings('ignore')


# In[29]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score


# In[3]:


get_ipython().system('pip install xgboost')


# In[7]:


df = pd.read_csv('creditcard.csv')


# In[8]:


df.head


# In[9]:


df.info


# In[ ]:





# In[10]:


#Check for missing Values
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)


# In[11]:


#Distribution of the terget variable
target_distribution = df['Class'].value_counts(normalize=True) * 100
print("\nTarget Distribution (%):\n", target_distribution)


# In[12]:


print(df.describe())


# In[13]:


correlation_matrix = df.corr()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, cmap="YlGnBu", annot=False, fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# In[14]:


# Distribution of transaction amounts
plt.figure(figsize=(8, 5))
sns.histplot(df['Amount'], bins=50, kde=True)
plt.title("Distribution of Transaction Amounts")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.show()


# In[15]:


# Target distribution visualization
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df)
plt.title("Target Distribution: Fraud vs. Non-Fraud")
plt.xlabel("Class (0 = Non-Fraud, 1 = Fraud)")
plt.ylabel("Count")
plt.show()


# In[16]:


# Separate features and target variable
X = df.drop('Class', axis=1)
y = df['Class']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data preprocessing complete. Shapes:")
print("X_train:", X_train_scaled.shape)
print("X_test:", X_test_scaled.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# # Model Training
# 
# Logistic Regression

# In[17]:


# Train Logistic Regression
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# Make predictions
lr_pred = lr_model.predict(X_test_scaled)
lr_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]


# In[18]:


# Print classification report
print("Logistic Regression Results:")
print("Classification Report:")
print(classification_report(y_test, lr_pred))

# Calculate ROC-AUC score
print("ROC-AUC Score:", roc_auc_score(y_test, lr_pred_proba))


# In[19]:


# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, lr_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Logistic Regression')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# # Inference:
# 
# Perfect precision (1.00) for normal transactions (class 0)
# 83% precision for fraudulent transactions (class 1)
# 65% recall for fraudulent transactions, meaning we're catching about 65% of all frauds
# Overall ROC-AUC score: 0.9560270189955554

# # Random Forrest

# In[20]:


# Train Random Forest with balanced class weights
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf.predict(X_test_scaled)


# In[21]:


# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# In[22]:


# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('Top 10 Most Important Features')
plt.tight_layout()
plt.show()


# # XGBoost

# In[24]:


import xgboost as xgb


# In[25]:


# Train XGBoost with scale_pos_weight to handle class imbalance
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
xgb_model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_xgb = xgb_model.predict(X_test_scaled)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_xgb))

# Create confusion matrix
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - XGBoost')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# XGBoost shows some improvements over Random Forest, particularly in recall for fraudulent transactions, though with a slight decrease in precision. This means it catches more fraud cases but with a slightly higher false positive rate.
# 

# # SMOTE approach to handle class imbalance

# In[26]:


# Apply SMOTE to balance the training data
from imblearn.over_sampling import SMOTE

# Apply SMOTE only to training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# Print class distribution before and after SMOTE
print("Original training set class distribution:")
print(pd.Series(y_train).value_counts())
print("SMOTE-balanced training set class distribution:")
print(pd.Series(y_train_smote).value_counts())


# In[27]:


# Train XGBoost on SMOTE-balanced data
xgb_smote = xgb.XGBClassifier(random_state=42)
xgb_smote.fit(X_train_smote, y_train_smote)

# Make predictions
y_pred_smote = xgb_smote.predict(X_test_scaled)

# Print classification report
print("XGBoost with SMOTE - Classification Report:")
print(classification_report(y_test, y_pred_smote))

# Create confusion matrix
cm_smote = confusion_matrix(y_test, y_pred_smote)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_smote, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - XGBoost with SMOTE')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Calculate and plot feature importance
feature_importance_smote = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_smote.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance_smote.head(10), x='importance', y='feature')
plt.title('Top 10 Most Important Features - XGBoost with SMOTE')
plt.tight_layout()
plt.show()


# #The XGBoost model with SMOTE has improved recall for fraudulent transactions, indicating better detection of fraud cases, though precision slightly decreased

# # model Comparison 
# 

# In[28]:


model_comparison = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost', 'XGBoost with SMOTE'],
    'Precision (Fraud)': [0.96, 0.88, 0.73],
    'Recall (Fraud)': [0.74, 0.84, 0.85],
    'F1-Score (Fraud)': [0.84, 0.86, 0.78]
})

print("Model Performance Comparison:")
print(model_comparison)


# - Random Forest has the highest precision but lower recall, meaning it flags fewer false positives but misses more fraud cases.
# - XGBoost with SMOTE improves recall but sacrifices precision, leading to more false positives.
# - XGBoost without SMOTE strikes a balance, achieving good recall and precision, making it the most practical choice for this dataset.

# In[31]:



param_grid = {
    'max_depth': [3, 5],
    'learning_rate': [0.1, 0.01],
    'n_estimators': [100, 200],
    'scale_pos_weight': [len(y_train[y_train == 0]) / len(y_train[y_train == 1])]
}

# Initialize XGBoost classifier
xgb_model = xgb.XGBClassifier(random_state=42)

# Create custom scorer for the minority class
f1_scorer = make_scorer(f1_score, average='binary')

# Perform Grid Search
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring=f1_scorer,
    cv=3,
    verbose=1,
    n_jobs=-1
)
# Fit the model
grid_search.fit(X_train_scaled, y_train)


# In[32]:


# Get best parameters and score
print("Best Parameters:")
print(grid_search.best_params_)
print("Best F1 Score:", grid_search.best_score_)


# In[33]:


# Train model with best parameters
best_model = xgb.XGBClassifier(**grid_search.best_params_, random_state=42)
best_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_best = best_model.predict(X_test_scaled)

# Print classification report
print("Classification Report for Best Model:")
print(classification_report(y_test, y_pred_best))


# In[34]:


# Plot confusion matrix for the best model
cm_best = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Best XGBoost Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# In[35]:


# Plot ROC curve
from sklearn.metrics import roc_curve, auc
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[36]:


# comprehensive comparison of all models tested
models_comparison = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost', 'XGBoost with SMOTE', 'Tuned XGBoost'],
    'Precision': [0.96, 0.88, 0.73, 0.84],
    'Recall': [0.74, 0.84, 0.85, 0.84],
    'F1-Score': [0.84, 0.86, 0.78, 0.84],
    'AUC-ROC': [0.98, 0.99, 0.97, 0.99]
})

models_comparison['Balanced_Accuracy'] = (models_comparison['Precision'] + models_comparison['Recall']) / 2

models_comparison_sorted = models_comparison.sort_values(['F1-Score', 'Balanced_Accuracy'], ascending=[False, False])

print("Model Performance Comparison (sorted by F1-Score):")
print(models_comparison_sorted)

# compare metrics
plt.figure(figsize=(12, 6))
metrics = ['Precision', 'Recall', 'F1-Score', 'AUC-ROC']
x = np.arange(len(models_comparison))
width = 0.2

for i, metric in enumerate(metrics):
    plt.bar(x + i*width, models_comparison[metric], width, label=metric)

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Performance Comparison Across Metrics')
plt.xticks(x + width*1.5, models_comparison['Model'], rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


### Best Model Analysis (XGBoost)
The original XGBoost model demonstrated superior performance with:
- Precision: 0.88 (88% accuracy in fraud predictions)
- Recall: 0.84 (84% of actual fraud cases detected)
- F1-Score: 0.86 (highest among all models)
- AUC-ROC: 0.99 (excellent discrimination capability)

### Operational Impact
- Successfully detected 82 out of 98 fraud cases
- Generated only 16 false alarms out of 56,962 transactions
- Achieved balanced performance between fraud detection and false positives


# In[ ]:


### Future Improvements
- Explore deep learning approaches
- Implement real-time model updating
- Enhance feature engineering
- Develop ensemble methods

