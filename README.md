# 🚗 Used Car Price Prediction — Machine Learning

A machine learning project that predicts the **price of used cars** based on vehicle characteristics such as brand, model, model year, mileage, fuel type, engine, transmission, colors, accident history, and title information.

The project uses **Random Forest Regression** to learn patterns from historical used-car data and estimate the expected market price of a vehicle.

---

## 📌 Project Overview

Used car prices depend on multiple factors, including the age and mileage of the vehicle, brand, model, engine specifications, transmission, fuel type, and accident history.

In this project, a **Random Forest Regressor** is trained to predict used-car prices from these features.

### Objective

> Build a regression model capable of predicting used-car prices with reasonable accuracy using vehicle-related features.

---

## 📊 Dataset

The dataset contains information about used vehicles and their selling prices.

### Features

| Feature        | Description                      |
| -------------- | -------------------------------- |
| `brand`        | Vehicle manufacturer             |
| `model`        | Vehicle model                    |
| `model_year`   | Manufacturing/model year         |
| `milage`       | Vehicle mileage                  |
| `fuel_type`    | Type of fuel used                |
| `engine`       | Engine specifications            |
| `transmission` | Transmission type                |
| `ext_col`      | Exterior color                   |
| `int_col`      | Interior color                   |
| `accident`     | Accident/damage history          |
| `clean_title`  | Clean title information          |
| `price`        | Target variable — used car price |

Additional engineered features were created during preprocessing, including:

* `car_age`
* `horsepower`
* `engine_liters`
* `has_accident`
* `clean_title_flag`

---

## 🔧 Data Preprocessing

The dataset required preprocessing before training the model.

### Steps performed

* Loaded the dataset using Pandas
* Checked dataset structure and missing values
* Converted `price` from currency-formatted strings to numerical values
* Converted `milage` from text such as `"51,000 mi."` into numerical values
* Created vehicle age from `model_year`
* Extracted horsepower and engine-liter information from the `engine` column
* Converted accident information into a numerical feature
* Converted clean-title information into a numerical feature
* Handled missing numerical values using median imputation
* Handled missing categorical values using most-frequent imputation
* Applied One-Hot Encoding to categorical features
* Removed extreme price outliers
* Split the dataset into training and testing sets

---

## 🤖 Machine Learning Model

### Random Forest Regressor

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

## 📊 Actual vs Predicted Prices

The model predictions were compared with the actual prices in the test dataset to evaluate how closely the predictions followed the real values.

```python
plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Used Car Prices")

plt.show()
```

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Scikit-learn**
* **Jupyter Notebook**

### Machine Learning

* Random Forest Regression
* Train-Test Split
* One-Hot Encoding
* Feature Engineering
* Data Imputation
* Regression Evaluation

---

## 📁 Project Structure

```text
Used-Car-Price-Prediction/
│
├── model.ipynb
├── used_cars.csv
├── README.md
└── requirements.txt
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the project

```bash
cd Used-Car-Price-Prediction
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

Open `model.ipynb` and run the cells.

---

## 🔮 Future Improvements

Potential improvements for the project include:

* Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
* Testing additional regression algorithms
* More advanced engine feature extraction
* Additional feature engineering
* Cross-validation
* Model deployment using Flask or FastAPI
* Building a web interface for price prediction

---

## 👩‍💻 Author

**Varsha A.**

Aspiring Data Scientist & Machine Learning Engineer

* GitHub: [Varshh-hub](https://github.com/Varshh-hub)

---

## ⭐ Project Highlight

**Random Forest Regression — R² Score: 73.42%**

This project demonstrates an end-to-end machine learning workflow, from **data cleaning and feature engineering to preprocessing, model training, evaluation, and prediction**.
