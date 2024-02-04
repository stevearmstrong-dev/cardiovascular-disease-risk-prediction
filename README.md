# Cardiovascular Disease Risk Prediction

## Table of Contents
- [Introduction and Motivation](#introduction-and-motivation)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Model Building and Training](#model-building-and-training)
- [Model Validation and Testing](#model-validation-and-testing)
- [Model Evaluation](#model-evaluation)
- [Visualizations](#visualizations)
- [Key Findings](#key-findings)
- [Report Breakdown](#report-breakdown)
- [Interpretation](#interpretation)
- [Impact](#impact)
- [Contact](#contact)
- [License](#license)



### Introduction and Motivation

This project aims to develop a robust predictive model for assessing the risk of cardiovascular disease (CVD) using the random forest algorithm. Cardiovascular diseases are among the leading causes of mortality globally. Early risk prediction can significantly improve outcomes through timely intervention and preventative measures. This project combines data exploration, visualization, and machine learning to provide insights into CVD risk factors and predict individual risk levels.

### Features

* Data Exploration and Visualization: Utilizing seaborn and Matplotlib for in-depth analysis and visualization of dataset attributes.
* Data Preprocessing: Employing Scikit-learn for data cleaning and preparation tasks.
* Predictive Modeling: Building and training a random forest classifier to accurately predict CVD risk.

### Dataset
The dataset comprises variables such as age, gender, blood pressure, cholesterol levels, and other significant CVD risk factors.

### Technologies Used
- Python
- Scikit-learn
- Seaborn
- Matplotlib

### Model Building and Training

The RandomForestClassifier from sklearn was used to create a predictive model, with `n_estimators` specifying the number of trees in the forest. The model was trained using `X_train` for the input features and `y_train` for the target variable, ensuring a robust learning process.

<img width="423" alt="Screenshot 2024-02-03 at 10 49 57 PM" src="https://github.com/W0474997SteveArmstrong/cardiovascular-disease-risk-prediction/assets/113034949/9ce8d730-c9db-4b33-ad12-45459ab25342">


### Model Validation and Testing
- **Prediction**: Applied the trained model to the testing dataset to predict cardiovascular disease risk.
- **Probability Assessment**: Computed prediction probabilities, offering insights into model confidence levels for each prediction.

### Model Evaluation
- **Classification Report**: Generated a detailed classification report using sklearn, providing precision, recall, and f1-score metrics for a comprehensive performance assessment.
- **ROC Score**: Calculated the receiver operating characteristic (ROC) score, measuring the model's ability to distinguish between classes.

### PairGrid - Histogram and ScatterPlot

![PairGrid Histogram and ScatterPlot](https://github.com/W0474997SteveArmstrong/cardiovascular-disease-risk-prediction/assets/113034949/41717290-3e05-48cb-96b8-09f6917fc4c5)

### Distribution of the Categorical Features of the Dataset

![Distribution of Categorical Features](https://github.com/W0474997SteveArmstrong/cardiovascular-disease-risk-prediction/assets/113034949/873e9f70-e0d8-4096-be85-d56ca9003ba5)

### Distribution of the Numerical Features of the Dataset

![Distribution of Numerical Features](https://github.com/W0474997SteveArmstrong/cardiovascular-disease-risk-prediction/assets/113034949/4ebfe187-0126-46e4-be60-7ce04449cdb7)

### Visualization of Factors causing Heart Disease

![Factors Causing Heart Disease](https://github.com/W0474997SteveArmstrong/cardiovascular-disease-risk-prediction/assets/113034949/40e5fd17-5538-4ef3-a575-a8a1762004b4)


## Key Findings
- The RandomForestClassifier demonstrated promising accuracy in predicting cardiovascular disease risk, as evidenced by the classification report and ROC score.

<img width="605" alt="Screenshot 2024-02-03 at 10 25 55 PM" src="https://github.com/W0474997SteveArmstrong/cardiovascular-disease-risk-prediction/assets/113034949/63f12547-f63d-403d-8cf9-e88a97026817">

## Report Breakdown:

### Class 0 (Negative Class):
- **Precision**: 92% of instances predicted as class 0 are actually class 0.
- **Recall**: The model correctly identifies 100% of all actual class 0 instances.
- **F1-Score**: 96%, indicating a very high balance between precision and recall for class 0.
- **Support**: There are 85,134 actual instances of class 0 in the dataset.

### Class 1 (Positive Class):
- **Precision**: 48% of instances predicted as class 1 are actually class 1.
- **Recall**: Only 2% of the actual class 1 instances were correctly identified by the model.
- **F1-Score**: 5%, indicating a poor balance between precision and recall for class 1.
- **Support**: There are 7,523 actual instances of class 1 in the dataset.

### Accuracy:
- Overall, the model correctly predicted 92% of all cases. However, this metric can be misleading for imbalanced classes.

### Macro Avg:
- **Precision**: Average precision across both classes without considering class imbalance is 70%.
- **Recall**: Average recall across both classes is 51%.
- **F1-Score**: Average F1 score is 50%.

### Weighted Avg:
- Accounts for class imbalance by weighting the average based on the number of instances in each class.
- **Precision**: 88% considering class imbalance.
- **Recall**: Same as accuracy, 92%.
- **F1-Score**: 88%, considering class imbalance.


## Interpretation:

While the model performs exceptionally well on class 0 (likely the majority class), it struggles significantly with class 1, as indicated by the low recall and F1-score for class 1. This suggests the model is biased towards the majority class and has difficulties identifying the minority class (class 1), which is a common issue in imbalanced datasets.

## Impact
- **Predictive Power**: This model significantly enhances my ability to predict cardiovascular disease risk, potentially informing more targeted preventative measures.
- **Model Confidence**: Probability assessments provide valuable insights into the model's confidence in its predictions, guiding clinical decision-making processes.

## Contact
For any questions or discussions, feel free to contact me at [steve@stevearmstrong.org](mailto:steve@stevearmstrong.org).

### License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.


