# Africa Population Trends and Health Analysis Using UN Dataset

## Project Overview

This project analyzes population, demographic, and health indicators across African countries using an open-source dataset derived from **United Nations (UN) population and health statistics**.

The main objective is to investigate how demographic factors such as fertility rate, life expectancy, and mortality indicators influence key development outcomes. The project applies data preprocessing, feature engineering, exploratory analysis, and supervised machine learning techniques to answer important population-health research questions.

The analysis focuses on building predictive models that can help understand relationships between:

- Fertility and maternal mortality
- Life expectancy and population growth
- Under-five mortality and life expectancy
- Demographic characteristics and health outcomes

The project demonstrates an end-to-end machine learning workflow, including:

- Data cleaning
- Missing value handling
- Feature engineering
- Model development
- Model evaluation
- Predictive analysis


---

# Project Objectives

The main objectives of this project are:

1. Analyze population and health trends across African countries.
2. Identify relationships between demographic variables and health outcomes.
3. Engineer meaningful features from raw demographic data.
4. Develop machine learning models for prediction tasks.
5. Compare different regression algorithms and evaluate their performance.
6. Generate insights that can support evidence-based decision-making.


---

# Dataset Description

The dataset used in this project is derived from publicly available **United Nations population and health indicators**.

The dataset contains demographic and health-related indicators for African countries, including:

| Category | Examples of Variables |
|---|---|
| Population | Population size, annual population growth rate |
| Fertility | Total fertility rate |
| Mortality | Maternal mortality ratio, under-five mortality |
| Health | Life expectancy at birth |
| Demographics | Age groups and population characteristics |


The dataset was cleaned and transformed before modeling to ensure consistency and reliability.


---

# Project Structure


Africa-Population-Trend-and-Health-Analysis/
│
├── data/
│ └── df_africa_cleaned.xls
│
├── notebooks/
│ └── main_analysis.
│
├── src/
│ ├── data_processor.py
│ ├── feature_engineer.py
│ ├── model_trainer.py
│ ├── model_evaluator.py
│ └── init.py
│
├── README.md
│
└── requirements.txt


---

# Technologies Used

## Programming Language

- Python 3.x


## Libraries

| Library | Purpose |
|---|---|
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Scikit-learn | Machine learning models |
| Matplotlib | Data visualization |
| Seaborn | Exploratory data visualization |


---

# Installation and Setup

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/Africa-Population-Trend-and-Health-Analysis.git

cd Africa-Population-Trend-and-Health-Analysis
2. Create Virtual Environment
python -m venv venv

Activate environment:

Windows
venv\Scripts\activate
Linux/Mac
source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
Running the Project

The main analysis can be executed using Jupyter Notebook:

jupyter notebook notebooks/main_analysis.ipynb

The notebook contains:

Data loading
Data cleaning
Feature engineering
Model training
Evaluation
Prediction analysis
Data Preprocessing

Before applying machine learning models, several preprocessing steps were performed.

Handling Missing Values

The dataset was examined for missing values.

The main missing variable identified was:

maternal_mortality_ratio_deaths_per_100000_population

Missing values were investigated against the original UN dataset to ensure data quality.

Missing values were handled through appropriate cleaning procedures before modeling.

Data Cleaning Steps

The preprocessing pipeline included:

Conversion of string missing values (NaN, nan) into proper null values.
Removal of incomplete observations.
Validation of dataset consistency.
Preparation of numerical features for machine learning.
Feature Engineering

Feature engineering was performed to improve model performance and create meaningful variables.

Fertility Category

A new categorical feature was created:

fertility_category

This feature classifies observations into:

High Fertility
Low Fertility

This feature was created to support classification analysis and investigate relationships between fertility patterns and health indicators.

Machine Learning Approach

The project uses supervised machine learning regression models.

Three major models were evaluated:

1. Linear Regression
Purpose

Linear Regression was used as a baseline model to understand linear relationships between demographic predictors and health outcomes.

Advantages
Easy interpretation
Fast training
Provides understanding of feature relationships
2. Ridge Regression
Purpose

Ridge Regression was selected because demographic variables often have strong correlations with each other.

The regularization component helps reduce overfitting and improves model stability.

Advantages
Handles multicollinearity
Improves generalization
Works well with correlated demographic features
3. Random Forest Regression
Purpose

Random Forest was selected because population and health relationships are often complex and nonlinear.

The model captures interactions between multiple demographic factors.

Advantages
Handles nonlinear relationships
Robust against noise
Provides feature importance analysis
Research Hypotheses Tested
Hypothesis 1
Fertility Predicts Maternal Mortality

Research Question:

Can fertility-related indicators predict maternal mortality rates?

Target variable:

maternal_mortality_ratio_deaths_per_100000_population

Problem Type:

Regression

Hypothesis 2
Life Expectancy Predicts Population Growth

Research Question:

Can life expectancy indicators explain population growth patterns?

Target variable:

population_annual_rate_of_increase_percent

Problem Type:

Regression

Hypothesis 3
Under-Five Mortality Predicts Life Expectancy

Research Question:

Can child mortality indicators predict national life expectancy?

Target variable:

life_expectancy_at_birth_for_both_sexes_years

Problem Type:

Regression

Model Evaluation

Models are evaluated using regression performance metrics such as:

Mean Absolute Error (MAE)

Measures the average prediction error.

Lower values indicate better performance.

Mean Squared Error (MSE)

Measures squared prediction errors and penalizes large mistakes.

Root Mean Squared Error (RMSE)

Provides error magnitude in the original target variable scale.

R² Score

Measures how much variation in the target variable is explained by the model.

Higher values indicate better predictive performance.

Key Findings

The analysis demonstrates that:

Demographic indicators provide valuable information for predicting health outcomes.
Fertility, mortality, and life expectancy variables are strongly interconnected.
Machine learning models can identify complex relationships within population-health datasets.
Tree-based models such as Random Forest are useful when relationships are nonlinear.
Future Improvements

Future work could include:

1. More Advanced Models

Experiment with:

Gradient Boosting
XGBoost
LightGBM
Neural Networks
2. Time-Series Analysis

Analyze changes in African population indicators over multiple years using:

ARIMA models
Prophet
LSTM networks
3. Interactive Dashboard

Create a visualization dashboard using:

Streamlit
Power BI
Tableau
4. Geographic Analysis

Add spatial analysis using:

Country-level maps
GIS visualization
Regional comparisons
Contributors

Jacob Ajak Makuach Abuol

Master's Student in Information Technology
Carnegie Mellon University Africa

License

This project is intended for educational and research purposes.

Dataset sources should be properly acknowledged according to the original UN data licensing terms.

Acknowledgements
United Nations Population and Health Statistics
Open-source Python machine learning community
Carnegie Mellon University Africa