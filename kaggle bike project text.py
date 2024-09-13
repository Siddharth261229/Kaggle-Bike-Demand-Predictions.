# Analysis Notes

### Predicted Variable (`demand`) Distribution:
- The target variable, **demand**, is **not normally distributed**. This issue will be addressed through transformation techniques like log normalization to stabilize variance.

### Features to be Dropped:
Based on the preliminary analysis, the following features are irrelevant or highly correlated with other variables and will be dropped:
- `weekday`
- `year`
- `workingday`
- `atemp`
- `windspeed`

### Correlations and Insights:
- **Temperature and Demand**: There appears to be a **direct correlation** between temperature and demand. As the temperature increases, the demand for bikes tends to rise.
- **Temperature (`temp`) vs. "Feels-like" Temperature (`atemp`)**: The plots for `temp` and `atemp` are nearly identical, suggesting strong multicollinearity between these two features.
- **Humidity and Windspeed**: These variables seem to influence bike demand. However, further **statistical analysis** is required to confirm the nature and strength of their effects.

### Autocorrelation:
- The **demand** feature exhibits **high autocorrelation**, meaning the current bike demand is strongly related to past values. This insight will be useful for creating lag variables to capture time dependency in the model.
