# GESTATIONAL BLOOD SUGAR TRACKER (GBST)

GBST is an AI based mobile application tailored towards Healthcare Informatics. GBST model is a supervised classifier machine learning model for predicting blood sugar levels in pregnant women. I ensured the dataset was clean to enable a more accurate model score. I implemented LazyClassifier from LazyPredict which offers a quick assessment of various machine learning models. This initial exploration provided valuable insights into potential model choices, guiding subsequent decisions in the model-building process. A Gaussian Naive Bayes classifier was selected for its suitability in predicting blood sugar levels. Through meticulous evaluation involving the splitting of the dataset into training and testing sets, the model's accuracy was rigorously assessed, providing a quantifiable measure of its predictive capability. 84% accuracy score was obtained. To ensure the model's re-usability and deployment, it was serialized using the pickle library.  This is to enable its seamless integration and interactivity with client-side applications like web or mobile applications.
A restful API was built to facilitate interaction with the client side. The client side was built using Expo React Native which is an Integrated Development Environment that enables a  cross platform building of mobile applications on different architectures e.g android and IOS.
Below are the github repositories for the project:
GBST Model - https://github.com/Inioluwajeremiah/gbst_ml_model 
GBST Restful API - https://github.com/Inioluwajeremiah/gbst-api 
GBST React Native - https://github.com/Inioluwajeremiah/gbstApp.

```python
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
gdm_dataset = pd.read_excel('docs\gdm_dataset.xlsx')

# Clean the dataset by dropping rows with any empty values
gdm_clean_dataset = gdm_dataset.dropna(how='any').reset_index(drop=True)

# Save the cleaned dataset to a CSV file
gdm_clean_dataset.to_csv('docs/gdm_clean_dataset_reset_index.csv', index=False)

# Extract relevant columns for further analysis
result = gdm_clean_dataset[['Prediabetes', 'Class Label(GDM /Non GDM)']]

# Modify 'Prediabetes' values based on conditions
result_col = []
for i in range(0, len(result['Prediabetes'])):
    if (gdm_clean_dataset['Prediabetes'][i] == 1 and gdm_clean_dataset['Class Label(GDM /Non GDM)'][i] == 1):
        result_col.append(gdm_clean_dataset['Prediabetes'][i])
    elif (gdm_clean_dataset['Prediabetes'][i] == 0 and gdm_clean_dataset['Class Label(GDM /Non GDM)'][i] == 1):
        result_col.append(2)
    else:
        result_col.append(gdm_clean_dataset['Class Label(GDM /Non GDM)'][i])

# Create a new DataFrame with modified 'Prediabetes' values
final_df = gdm_clean_dataset.iloc[0:gdm_dataset.shape[0], 0:gdm_dataset.shape[1] - 2]
final_df['output'] = result_col

# Save the final dataset to a CSV file
final_df.to_csv('docs/final_gdm_clean_dataset.csv', index=False)

# Read the final dataset from the CSV file
gdm_clean_dataset = pd.read_csv('docs/final_gdm_clean_dataset.csv')

# Extract target variable 'output' and features 'X'
target_y = gdm_clean_dataset[['output']]
X = gdm_clean_dataset.iloc[:, 0:-1]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, target_y, test_size=.3, random_state=123)

# Build a LazyClassifier model
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# Display the list of models
models

# Build a Gaussian Naive Bayes model
gaussianNB = GaussianNB()

# Split the data into training and testing sets
X1_train, X1_test, y1_train, y1_test = train_test_split(X, target_y, test_size=.3, random_state=123)

# Train the Gaussian Naive Bayes classifier
gaussianNB.fit(X1_train, y1_train)

# Make predictions on the testing data
y_predict = gaussianNB.predict(X1_test)

# Check the accuracy score of the model on the testing dataset
accuracy_score = accuracy_score(y1_test, y_predict)
print("Accuracy score:", accuracy_score)

# Serialize the model using pickle
pickle.dump(gaussianNB, open('model.pkl', 'wb'))
```
