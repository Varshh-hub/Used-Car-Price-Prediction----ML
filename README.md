# Used Car Price Prediction — ML

A machine learning project that predicts the **price of used cars** based on vehicle characteristics such as brand, model, model year, mileage, fuel type, engine, transmission, colors, accident history, and title information.

The project uses **Random Forest Regression** to learn patterns from historical used-car data and estimate the expected market price of a vehicle.

---

## Project Overview

Used car prices depend on multiple factors, including the age and mileage of the vehicle, brand, model, engine specifications, transmission, fuel type, and accident history.

In this project, a **Random Forest Regressor** is trained to predict used-car prices from these features.

### Objective

> Build a regression model capable of predicting used-car prices with reasonable accuracy using vehicle-related features.

---

## 🤖 Machine Learning Model

The project uses a **Random Forest Regressor**, an ensemble learning algorithm that combines multiple decision trees to produce a more robust prediction.

```python
RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
```

The preprocessing and model were combined into a Scikit-learn pipeline.

---

## 📈 Model Performance

The Random Forest model achieved the following results on the test dataset:

| Metric            |          Score |
| ----------------- | -------------: |
| **MAE**           |  **$8,823.70** |
| **RMSE**          | **$17,685.25** |
| **R² Score**      |     **0.7342** |
| **R² Percentage** |     **73.42%** |

### Interpretation

* **MAE:** The model's predictions differ from the actual car price by approximately **$8,823.70 on average**.
* **RMSE:** The RMSE of **$17,685.25** indicates the overall magnitude of prediction errors, with larger errors receiving more weight.
* **R²:** The model explains approximately **73.42% of the variation in used-car prices** in the test dataset.

---

## 👩‍💻 Author

**Varsha A.**
AI & ML Graduate | Junior Data Scientist & Machine Learning Engineer | Python | SQL | Excel | Power BI | Prompt Engineer | Front-End Developer

---
