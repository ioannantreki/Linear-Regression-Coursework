# Linear-Regression-Coursework
Academic Machine Learning coursework focusing on Linear Regression. Includes Linear Algebra, Python implementations, and model evaluations using the Boston Housing dataset.

## 📚 Project Overview
- **Vector and Matrix Operations**: Using NumPy for computations like dot products, norms, and matrix multiplication, with validation through handwritten calculations.
- **Linear Regression**: Deriving the Least Squares estimator for the multivariate case, implementing and training Linear Regression models from scratch using Python.
- **Model Evaluation**: Analyzing performance using metrics such as R² and Residual Standard Error (RSE), exploring the effects of different train-test splits (80/20, 70/30, and 50/50).
- **Optional Gradient Computation**: Analytical derivations for functions with vector/matrix inputs.

## 📊 Dataset
The project uses the Boston Housing Dataset, a popular benchmark dataset for regression tasks. It contains information about housing prices in Boston suburbs based on 13 features (e.g., crime rate, average number of rooms, property tax rate). The target variable is the Median value of owner-occupied homes (in $1000s).

## ✨ Features
- CRIM: Per capita crime rate by town.
- ZN: Proportion of residential land zoned for lots over 25,000 sq. ft.
- INDUS: Proportion of non-retail business acres per town.
- CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
- NOX: Nitric oxide concentration (parts per 10 million).
- RM: Average number of rooms per dwelling.
- AGE: Proportion of owner-occupied units built prior to 1940.
- DIS: Weighted distances to five Boston employment centers.
- RAD: Index of accessibility to radial highways.
- TAX: Full-value property tax rate per $10,000.
- PTRATIO: Pupil-teacher ratio by town.
- B: 1000(Bk – 0.63)², where Bk is the proportion of people of African American descent by town.
- LSTAT: Percentage of lower status of the population.
  
- **Target Variable (y)**: MEDV: Median value of owner-occupied homes in $1000’s.

**Notes**:
- Due to privacy and licensing reasons, the dataset is not included in this repository.
- The dataset is included in the Scikit-learn library and can be loaded programmatically.
- If the raw dataset (boston_housing.csv) is not provided, you can load it using:

```python
from sklearn.datasets import load_boston
import pandas as pd

data = load_boston()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MEDV'] = data.target
```
## 📚 Results
The table below summarizes the model's performance across different train-test splits, evaluated using Root Mean Squared Error (RMSE) and R² metrics:

| Train/Test Split | Training RMSE | Training R² | Testing RMSE | Testing R² |
|------------------|---------------|-------------|--------------|------------|
| 90/10           | 4.86          | 0.73        | 3.81         | 0.77       |
| 80/20           | 4.73          | 0.74        | 5.00         | 0.66       |
| 70/30           | 4.85          | 0.73        | 4.67         | 0.71       |
| 50/50           | 4.58          | 0.76        | 5.04         | 0.69       |

## 🚀 How to Use
1. **Clone the repository**:
```bash
git clone https://github.com/ioannantreki/Linear-Regression-Coursework.git
cd ML-Linear-Regression-Coursework
```
2. **Install Depedencies**: Use the following command to install the required Python libraries:
```bash
pip install numpy, matplotlib.pyplot, pandas, sklearn.model_selection
```
3. **Usage**: Navigate to the cloned directory in your terminal and execute the Linear Regression model script by running:
`linear_regression.py`

## ℹ️ Contact
For any inquiries or collaboration requests, please reach out via GitHub or email at ioannadreki31@gmail.com.







