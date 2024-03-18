## ***Heart Disease Prediction***

This project aims to predict the likelihood of heart disease using machine learning techniques. Cardiovascular diseases are a leading cause of mortality worldwide, responsible for a significant portion of global deaths. Early detection and management of heart disease can significantly reduce its fatality rate.

### Objective:
The objective of this project is to develop machine learning models to predict the chances of heart disease based on various relevant features. By analyzing and understanding these features, we can gain insights into the factors contributing to heart disease and improve early detection and management.

### Dataset:
The dataset used for this project is sourced from Kaggle: [Heart Disease Prediction](https://www.kaggle.com/datasets/ritwikb3/heart-disease-cleveland). It comprises 14 features that include demographic information, medical history, and clinical measurements. These features are described as follows:

1. **age:** Age in years
2. **sex:** Gender (1 = male, 0 = female)
3. **cp:** Chest pain type
   - 0: Typical angina
   - 1: Atypical angina
   - 2: Non-anginal pain
   - 3: Asymptomatic
4. **trestbps:** Resting blood pressure (mm Hg)
5. **chol:** Serum cholesterol (mg/dl)
6. **fbs:** Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
7. **restecg:** Resting electrocardiographic results
   - 0: Normal
   - 1: ST-T wave abnormality
   - 2: Probable or definite left ventricular hypertrophy
8. **thalach:** Maximum heart rate achieved
9. **exang:** Exercise-induced angina (1 = yes, 0 = no)
10. **oldpeak:** ST depression induced by exercise relative to rest
11. **slope:** Slope of the peak exercise ST segment
    - 0: Upsloping
    - 1: Flat
    - 2: Downsloping
12. **ca:** Number of major vessels colored by fluoroscopy (0-3)
13. **thal:** Thalassemia
    - 0: Normal
    - 1: Fixed defect
    - 2: Reversible defect
14. **condition:** Target variable indicating presence (1) or absence (0) of heart disease

### Methodology:
The project follows these key steps:
1. **Understanding the Problem:** Analyzing the importance of early detection and management of heart disease.
2. **Reading and Understanding the Data:** Loading and exploring the dataset to gain insights into its structure and features.
3. **Exploratory Data Analysis (EDA) and Visualization:** Analyzing the relationships between different features and their distributions.
4. **Modeling:** Developing machine learning models including Support Vector Machine (SVM), Logistic Regression, Decision Tree, and Random Forest to predict heart disease.
5. **Generate Insights:** Evaluating model performance, identifying important features, and drawing conclusions to aid in heart disease prediction and management.

### Libraries Used:
- NumPy: For numerical operations
- Pandas: For data manipulation
- Matplotlib and Seaborn: For data visualization
- Scikit-learn: For machine learning models and evaluation metrics

By leveraging machine learning techniques on this dataset, we aim to contribute to the early detection and management of heart disease, thereby reducing its impact on global health.
