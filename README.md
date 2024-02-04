# Cardiovascular Disease Risk Prediction

### Introduction and Motivation

This project aims to develop a robust predictive model for assessing the risk of cardiovascular disease (CVD) using the random forest algorithm. Cardiovascular diseases are among the leading causes of mortality globally. Early risk prediction can significantly improve outcomes through timely intervention and preventative measures. This project combines data exploration, visualization, and machine learning to provide insights into CVD risk factors and predict individual risk levels.

### Features

* Data Exploration and Visualization: Utilizing seaborn and Matplotlib for in-depth analysis and visualization of dataset attributes.
* Data Preprocessing: Employing Scikit-learn for data cleaning and preparation tasks.
* Predictive Modeling: Building and training a random forest classifier to accurately predict CVD risk.

### Dataset
The dataset comprises variables such as age, gender, blood pressure, cholesterol levels, and other significant CVD risk factors.

### Model Building and Training

The RandomForestClassifier from sklearn was used to create a predictive model, with `n_estimators` specifying the number of trees in the forest. The model was trained using `X_train` for the input features and `y_train` for the target variable, ensuring a robust learning process.

<img width="423" alt="Screenshot 2024-02-03 at 10 49 57 PM" src="https://github.com/W0474997SteveArmstrong/cardiovascular-disease-risk-prediction/assets/113034949/9ce8d730-c9db-4b33-ad12-45459ab25342">


### Model Validation and Testing
- **Prediction**: Applied the trained model to the testing dataset to predict cardiovascular disease risk.
- **Probability Assessment**: Computed prediction probabilities, offering insights into model confidence levels for each prediction.

### Model Evaluation
- **Classification Report**: Generated a detailed classification report using sklearn, providing precision, recall, and f1-score metrics for a comprehensive performance assessment.
- **ROC Score**: Calculated the receiver operating characteristic (ROC) score, measuring the model's ability to distinguish between classes.

<img width="612" alt="Screenshot 2024-02-03 at 10 50 53 PM" src="https://github.com/W0474997SteveArmstrong/cardiovascular-disease-risk-prediction/assets/113034949/a065e074-6e94-4f33-8b59-b61ac13e129a">


### PairGrid - Histogram and ScatterPlot

![PairGrid Histogram and ScatterPlot](https://github.com/W0474997SteveArmstrong/cardiovascular-disease-risk-prediction/assets/113034949/41717290-3e05-48cb-96b8-09f6917fc4c5)

### Distribution of the Categorical Features of the Dataset

![Distribution of Categorical Features](https://github.com/W0474997SteveArmstrong/cardiovascular-disease-risk-prediction/assets/113034949/873e9f70-e0d8-4096-be85-d56ca9003ba5)

### Distribution of the Numerical Features of the Dataset

![Distribution of Numerical Features](https://github.com/W0474997SteveArmstrong/cardiovascular-disease-risk-prediction/assets/113034949/4ebfe187-0126-46e4-be60-7ce04449cdb7)

### Visualization of Factors causing Heart Disease

![Factors Causing Heart Disease](https://github.com/W0474997SteveArmstrong/cardiovascular-disease-risk-prediction/assets/113034949/40e5fd17-5538-4ef3-a575-a8a1762004b4)




### License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.


