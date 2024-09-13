# README

## Bike Sharing Demand Prediction

This project is focused on predicting the demand for bike-sharing services using a dataset containing hourly rental data. The dataset includes various features such as weather conditions, date and time, and categorical variables like season, holiday, and weekday. Multiple linear regression is applied to model the relationship between these variables and the target variable, **demand**.

### Project Workflow

1. **Data Loading and Preprocessing**:
   - Load the dataset (`hour.csv`) using `pandas`.
   - Drop unnecessary columns: `['index', 'date', 'casual', 'registered']` to avoid redundancy.
   - Check for missing values and clean the data.

2. **Data Visualization**:
   - Visualize continuous variables like `temp`, `atemp`, `humidity`, and `windspeed` against the `demand` variable using scatter plots.
   - Use bar charts to visualize the average demand across categorical variables like `season`, `month`, `weekday`, `holiday`, `year`, `hour`, `workingday`, and `weather`.

3. **Outlier Detection**:
   - Basic statistics of the target variable `demand` are calculated.
   - Quantile values are used to detect possible outliers.

4. **Feature Selection**:
   - Correlation matrix is used to assess the linearity between numerical features and the `demand`.
   - Irrelevant features like `weekday`, `year`, `workingday`, `atemp`, and `windspeed` are removed.

5. **Autocorrelation Check**:
   - Autocorrelation in the demand is checked using the `plt.acorr()` function.
   - Lag features (`t-1`, `t-2`, `t-3`) are created to capture time dependency in the demand variable.

6. **Log Transformation**:
   - Log normalization is applied to the `demand` variable to stabilize variance and reduce skewness.

7. **Creating Dummy Variables**:
   - Categorical features (`season`, `holiday`, `weather`, `month`, `hour`) are converted to dummy variables using `pd.get_dummies()` to avoid the dummy variable trap.

8. **Train-Test Split**:
   - A 70-30 split is used to divide the dataset into training and testing sets.

9. **Modeling**:
   - A multiple linear regression model is built using the `sklearn.linear_model.LinearRegression` class.
   - The model is fitted on the training set, and performance is evaluated on both training and test sets using the R² score.

10. **Evaluation**:
    - The model's performance is evaluated using RMSE (Root Mean Square Error) and RMSLE (Root Mean Square Logarithmic Error) metrics.
    - The RMSLE is calculated to compare the accuracy of the model's predictions.

### Requirements

- Python 3.x
- Libraries:
  - `pandas`
  - `matplotlib`
  - `numpy`
  - `sklearn`

Install the required libraries using:

```bash
pip install pandas matplotlib numpy scikit-learn
```

### How to Run

1. Ensure you have the dataset `hour.csv` in the same directory as the script.
2. Run the script:
   ```bash
   python bikes_demand_prediction.py
   ```

### Dataset

The dataset used for this project is the `hour.csv` file, which contains the following key columns:

- **temp**: Temperature in Celsius.
- **atemp**: "Feels-like" temperature.
- **humidity**: Humidity level.
- **windspeed**: Wind speed.
- **season**: Season (Winter, Spring, Summer, Fall).
- **holiday**: Whether the day is a holiday or not.
- **workingday**: Whether the day is a working day.
- **demand**: Total bike rentals (target variable).

### Model Performance

- **R² on Train Set**: Indicates how well the model fits the training data.
- **R² on Test Set**: Indicates how well the model generalizes to unseen data.
- **RMSE**: Measures the average magnitude of the prediction error.
- **RMSLE**: Measures the accuracy in predicting demand when both actual and predicted values are log-transformed.

### Conclusion

The model uses multiple linear regression to predict bike-sharing demand based on various time and weather-related features. The preprocessing steps, feature selection, and transformation methods were carefully chosen to meet linear regression assumptions and improve model performance.
