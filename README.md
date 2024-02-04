# Cardiovascular Disease Risk Prediction

### Table of Contents
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
- [How to Use](#how-to-use)
- [Contact](#contact)
- [License](#license)



### Introduction and Motivation

This project aims to develop a robust predictive model for assessing the risk of cardiovascular disease (CVD) using the random forest algorithm. Cardiovascular diseases are among the leading causes of mortality globally. Early risk prediction can significantly improve outcomes through timely intervention and preventative measures. This project combines data exploration, visualization, and machine learning to provide insights into CVD risk factors and predict individual risk levels.

### Features

* Data Exploration and Visualization: Utilizing seaborn and Matplotlib for in-depth analysis and visualization of dataset attributes.
* Data Preprocessing: Employing Scikit-learn for data cleaning and preparation tasks.
* Predictive Modeling: Building and training a random forest classifier to accurately predict CVD risk.

### Technologies Used
- Python
- numpy
- pandas
- Scikit-learn
- Seaborn
- Matplotlib

### Dataset
The dataset comprises variables such as age, gender, blood pressure, cholesterol levels, and other significant CVD risk factors.

### Model Building and Training

The predictive model for cardiovascular disease risk prediction was constructed using the RandomForestClassifier from sklearn, leveraging its capabilities for handling complex datasets with a mix of categorical and numerical data.

#### Model Configuration
- **Algorithm**: RandomForestClassifier
- **Key Parameter**: `n_estimators` was set to specify the number of trees in the forest, chosen based on preliminary validation to balance between overfitting and computational efficiency.

#### Training Process
- **Training Data**: The model was trained using `X_train` for the input features, encompassing a diverse range of variables such as age, gender, blood pressure, and cholesterol levels.
- **Target Variable**: `y_train` represented the presence or absence of cardiovascular disease, serving as the output parameter for the model.
- **Methodology**: Employed a robust training methodology to ensure the model accurately captures the underlying patterns without overfitting to the training data.

![Model Training Visualization](https://github.com/W0474997SteveArmstrong/cardiovascular-disease-risk-prediction/assets/113034949/9ce8d730-c9db-4b33-ad12-45459ab25342)

The RandomForestClassifier was chosen for its efficacy in classification tasks, its intrinsic ability to manage overfitting, and its feature importance capabilities, which are instrumental for understanding the predictive power of the various risk factors involved in cardiovascular disease.

### Model Validation and Testing
- **Prediction**: Applied the trained model to the testing dataset to predict cardiovascular disease risk.
- **Probability Assessment**: Computed prediction probabilities, offering insights into model confidence levels for each prediction.

### Model Evaluation
- **Classification Report**: Generated a detailed classification report using sklearn, providing precision, recall, and f1-score metrics for a comprehensive performance assessment.
- **ROC Score**: Calculated the receiver operating characteristic (ROC) score, measuring the model's ability to distinguish between classes.

## Visualizations
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


## How to Use

This project is designed to be accessible and straightforward to run using Jupyter Notebooks, a popular tool in data science for interactive computing.

### Prerequisites

To run the `cvd_risk_prediction.ipynb` notebook, you'll need to have Python installed on your system along with Jupyter Notebook or JupyterLab. It's also recommended to use a virtual environment for Python projects to manage dependencies effectively.

### Installation

1. **Clone the Repository**: Start by cloning this repository to your local machine.
   ```bash
   git clone https://github.com/W0474997SteveArmstrong/cardiovascular-disease-risk-prediction.git
   cd cardiovascular-disease-risk-prediction

2. **Create a Virtual Environment (Optional but recommended)**:
   * For **conda** users:
     ``` bash
     conda create --name cvd_risk_prediction python=3.8
     conda activate cvd_risk_prediction
   * For **venv** users:
     ```bash
     python3 -m venv cvd_risk_prediction
     source cvd_risk_prediction/bin/activate  # On Windows use `cvd_risk_prediction\Scripts\activate`

3. **Install Required Packages**
   ```bash
   pip install numpy pandas jupyterlab matplotlib seaborn scikit-learn

4. **Running the Notebook**
   * Navigate to the Notebook Directory: Change directory to the `notebooks` folder.
     ```bash
     cd notebooks
   * Launch Jupyter Notebook
   ```bash
    jupyter notebook
5. **Open `cvd_risk_prediction.ipynb` in the Jupyter Notebook interface** and follow the instructions within the notebook to run the analyses.

### Understanding the Results
The notebook includes detailed comments and visualizations to help you understand each step of the process, from data exploration to model evaluation. Here's what to look for:

* Data Exploration and Visualization: Initial sections of the notebook provide insights into the dataset's structure and distribution of variables.
* Model Training: Look for the section where the RandomForestClassifier is trained with the cvd_cleaned.csv dataset.
* Model Evaluation: The final sections will show the model's performance on the test set, including accuracy, precision, recall, and the ROC score. Interpret these metrics to gauge the model's effectiveness in predicting cardiovascular disease risk.

## Contact
For any questions or discussions, feel free to contact me at [steve@stevearmstrong.org](mailto:steve@stevearmstrong.org).

### License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.


