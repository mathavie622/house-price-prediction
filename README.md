# House Price Prediction using Machine Learning

## 1. Project Title

**House Price Prediction using Machine Learning**

---

## 2. Project Overview

This project develops a Machine Learning model to predict house values based on different housing-related features.

The project uses the **California Housing dataset** and applies **Linear Regression** to predict the median house value.

The complete project was implemented using Python and Scikit-learn in Google Colab.

---

## 3. Objective

The main objective of this project is to build a Machine Learning regression model that can predict house values based on available housing features.

The model learns patterns from the training data and predicts the value of houses in the testing data.

---

## 4. Problem Statement

House prices depend on several factors such as income, house age, number of rooms, population, and geographical information.

Predicting house prices manually can be difficult because multiple factors affect the final value.

Therefore, a Machine Learning regression model can be used to estimate house values from historical housing data.

---

## 5. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab
- Machine Learning

---

## 6. Dataset

The project uses the **California Housing dataset** available through Scikit-learn.

The dataset contains housing-related numerical features and a target variable representing the median house value.

### Dataset Features

| Feature | Description |
|---|---|
| MedInc | Median income in the block |
| HouseAge | Median house age |
| AveRooms | Average number of rooms |
| AveBedrms | Average number of bedrooms |
| Population | Block population |
| AveOccup | Average number of household members |
| Latitude | Latitude of the block |
| Longitude | Longitude of the block |
| MedHouseVal | Median house value |

---

## 7. Dataset Exploration

The dataset was loaded into a Pandas DataFrame.

The following checks were performed:

- Dataset shape
- Column names
- Missing values
- Statistical description
- Sample records

These steps were performed to understand the structure and quality of the dataset before training the model.

---

## 8. Data Visualization

A scatter plot was created to visualize the relationship between median income and median house value.

```python
plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="MedInc",
    y="MedHouseVal",
    s=50
)

plt.title("House Price Prediction - Income vs House Value")
plt.xlabel("Median Income")
plt.ylabel("Median House Value")

plt.show()
```

The visualization helps observe the relationship between the selected input feature and house value.

---

## 9. Project Workflow

```text
California Housing Dataset
          ↓
Data Loading
          ↓
Data Exploration
          ↓
Missing Value Check
          ↓
Feature and Target Separation
          ↓
Train-Test Split
          ↓
Linear Regression Model
          ↓
House Value Prediction
          ↓
Model Evaluation
          ↓
Actual vs Predicted Visualization
          ↓
New House Value Prediction
```

---

## 10. Feature and Target Separation

The dataset was divided into input features and the target variable.

```python
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]
```

Here:

- `X` represents the input features.
- `y` represents the target house value.

The target column `MedHouseVal` was removed from the input features before model training.

---

## 11. Train-Test Split

The dataset was divided into training and testing sets.

```text
Training Data : 80%
Testing Data  : 20%
```

The training data was used to train the Linear Regression model, while the testing data was used to evaluate its performance.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

## 12. Machine Learning Model

### Linear Regression

The project uses **Linear Regression** as the Machine Learning algorithm.

Linear Regression is a supervised learning algorithm used for predicting continuous numerical values.

The model was created and trained using:

```python
model = LinearRegression()

model.fit(X_train, y_train)

print("Linear Regression model trained successfully!")
```

---

## 13. House Value Prediction

After training the model, predictions were generated using the testing data.

```python
y_pred = model.predict(X_test)

print("House Price Prediction completed successfully!")
print("First 10 Predicted Values:")
print(y_pred[:10])
```

The predicted values were then compared with the actual test values.

---

## 14. Model Evaluation

The model was evaluated using the following regression metrics:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

The metrics were calculated using:

```python
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)
```

---

## 15. Actual vs Predicted Values

A scatter plot was created to compare the actual house values with the values predicted by the model.

```python
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.6)

plt.xlabel("Actual House Values")
plt.ylabel("Predicted House Values")
plt.title("Actual vs Predicted House Values")

plt.show()
```

This visualization helps compare the model's predictions with the actual values.

---

## 16. New House Value Prediction

A sample house from the dataset was used to demonstrate prediction on a new input.

```python
sample_house = X.iloc[[0]].copy()

predicted_price = model.predict(sample_house)

print("Sample House Features:")
print(sample_house)

print("\nPredicted House Value:", round(predicted_price[0], 4))
```

The model generates a predicted house value based on the input features.

---

## 17. Final Result

The final result was generated using:

```python
print("=" * 55)
print("HOUSE PRICE PREDICTION - FINAL RESULT")
print("=" * 55)

print("Model Used        : Linear Regression")
print(f"Training Samples  : {len(X_train)}")
print(f"Testing Samples   : {len(X_test)}")
print(f"MAE               : {mae:.4f}")
print(f"MSE               : {mse:.4f}")
print(f"RMSE              : {rmse:.4f}")
print(f"R² Score          : {r2:.4f}")
print(f"Predicted Value   : {predicted_price[0]:.4f}")

print("\nProject Status: Completed Successfully! ✅")
```

The final output displays:

- Model Used
- Training Samples
- Testing Samples
- MAE
- MSE
- RMSE
- R² Score
- Predicted House Value
- Project Status

---

## 18. Results

The Linear Regression model was successfully trained to predict house values using the California Housing dataset.

The final evaluation metrics obtained from the test data are available in the project notebook.

The project demonstrates the use of Machine Learning for predicting continuous numerical values.

---

## 19. Advantages

- Simple and easy to implement
- Fast model training
- Suitable for regression problems
- Easy to interpret
- Can be used for continuous value prediction

---

## 20. Applications

House price prediction models can be used in:

- Real estate analysis
- Property valuation
- Housing market analysis
- Investment analysis
- Property recommendation systems

---

## 21. Limitations

- Model performance depends on the quality and features of the dataset.
- Linear Regression assumes a linear relationship between features and the target.
- Real-world house prices may depend on additional factors not included in the dataset.
- Predictions may vary when applied to data from different locations or markets.

---

## 22. Future Improvements

The project can be improved by:

- Comparing multiple regression algorithms
- Applying feature scaling
- Performing hyperparameter tuning
- Adding more relevant housing features
- Using advanced ensemble models
- Developing a web-based prediction interface
- Deploying the model as an application

---

## 23. Project Files

```text
house-price-prediction/
│
├── House_Price_Prediction_AI_ML.ipynb
└── README.md
```

### File Description

| File | Description |
|---|---|
| `House_Price_Prediction_AI_ML.ipynb` | Complete Machine Learning notebook |
| `README.md` | Project documentation |

---

## 24. Conclusion

The **House Price Prediction** project demonstrates a complete Machine Learning regression workflow.

The project includes:

- Dataset loading
- Dataset exploration
- Missing value checking
- Feature and target separation
- Train-test splitting
- Linear Regression model training
- House value prediction
- Model evaluation
- Regression metrics
- Actual vs predicted visualization
- New house value prediction

The project demonstrates how Machine Learning can be used to predict continuous house values from housing-related features.

---

## 25. Author

**Mathavi E**

Department of Electronics and Communication Engineering

GitHub: `mathavie622`
