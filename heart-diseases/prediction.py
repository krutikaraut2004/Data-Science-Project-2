# For prediction model , I chose = Random Forest Classifier (because of its high accuracy)
# importing required libraries
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# loading and reading the dataset
heart = pd.read_csv("heart_cleveland_upload.csv")

# creating a copy of dataset so that it will not affect our original dataset.
heart_df = heart.copy()

# Renaming some of the columns
heart_df = heart_df.rename(columns={'condition':'target'})

# model building
# Fixing our data in x and y. Here y contains target data and X contains rest all the features.
x = heart_df.drop(columns='target')
y = heart_df.target

# splitting our dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# feature scaling
scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.fit_transform(x_test)

# creating random forest classifier
model = RandomForestClassifier(n_estimators=20)
model.fit(x_train_scaler, y_train)
y_pred = model.predict(x_test_scaler)
accuracy = model.score(x_test_scaler, y_test)
print(accuracy)

print('Classification Report\n', classification_report(y_test, y_pred))
print('Accuracy: {}%\n'.format(round((accuracy_score(y_test, y_pred)*100),2)))

cm = confusion_matrix(y_test, y_pred)
print(cm)

# Creating a pickle file for the classifier
filename = 'heart-disease-prediction.pkl'
pickle.dump(model, open(filename, 'wb'))


#Purpose of the above code : 
# The purpose of the above code is to build a machine learning model using a Random Forest Classifier
# to predict heart disease based on a given dataset, and then save the trained model 
# for future use.