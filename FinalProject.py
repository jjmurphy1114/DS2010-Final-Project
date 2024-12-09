import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('movies_dataset.csv')
print(df.head())

# --- Conjecture 1: Genre Impact on Box Office Revenue ---

# One-hot encode the Genre column
genre_dummies = pd.get_dummies(df['Genre'], drop_first=True)

# Combine the one-hot encoded genres with the dataset
df = pd.concat([df, genre_dummies], axis=1)

# Select revenue as the target and genres as features
X_genre = genre_dummies
y_revenue = df['Revenue']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_genre, y_revenue, test_size=0.2, random_state=42)

# Train a Linear Regression model
model_genre = LinearRegression()
model_genre.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model_genre.predict(X_test)
print("Genre Model R2 Score:", r2_score(y_test, y_pred))

# Display feature importance
coef_genre = pd.DataFrame(model_genre.coef_, X_genre.columns, columns=['Coefficient'])
print("Genre Coefficients:\n", coef_genre)

# --- Conjecture 2: Ratings (Audience and Critical) Correlation with Revenue ---

# Select ratings as features and revenue as target
X_ratings = df[['AudienceRating', 'CriticRating']]
y_revenue = df['Revenue']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_ratings, y_revenue, test_size=0.2, random_state=42)

# Train a Linear Regression model
model_ratings = LinearRegression()
model_ratings.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model_ratings.predict(X_test)
print("Ratings Model R2 Score:", r2_score(y_test, y_pred))

# Display feature importance
coef_ratings = pd.DataFrame(model_ratings.coef_, X_ratings.columns, columns=['Coefficient'])
print("Ratings Coefficients:\n", coef_ratings)

# --- Conjecture 3: Budget Impact on Revenue with Diminishing Returns ---

# Use budget and its quadratic term as features
df['BudgetSquared'] = df['Budget'] ** 2
X_budget = df[['Budget', 'BudgetSquared']]
y_revenue = df['Revenue']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_budget, y_revenue, test_size=0.2, random_state=42)

# Train a Linear Regression model
model_budget = LinearRegression()
model_budget.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model_budget.predict(X_test)
print("Budget Model R2 Score:", r2_score(y_test, y_pred))

# Display feature importance
coef_budget = pd.DataFrame(model_budget.coef_, X_budget.columns, columns=['Coefficient'])
print("Budget Coefficients:\n", coef_budget)

# Plot the actual vs predicted revenue
plt.scatter(df['Budget'], df['Revenue'], label='Actual Data', alpha=0.6)
plt.plot(df['Budget'], model_budget.predict(X_budget), color='red', label='Model Prediction')
plt.xlabel('Budget')
plt.ylabel('Revenue')
plt.legend()
plt.title('Budget vs Revenue')
plt.show()

# --- Predictions for New Data ---

# Example new data for predictions
new_data = {
    'Genre': ['Action', 'Drama'],  # One-hot encode genres
    'AudienceRating': [85, 70],
    'CriticRating': [80, 60],
    'Budget': [100_000_000, 20_000_000]
}

new_df = pd.DataFrame(new_data)
new_df = pd.concat([new_df, pd.get_dummies(new_df['Genre'], drop_first=True)], axis=1)

# Predictions
genre_prediction = model_genre.predict(new_df[genre_dummies.columns])
ratings_prediction = model_ratings.predict(new_df[['AudienceRating', 'CriticRating']])
budget_prediction = model_budget.predict(new_df[['Budget', 'BudgetSquared']])

print("Predicted Revenue from Genre Model:", genre_prediction)
print("Predicted Revenue from Ratings Model:", ratings_prediction)
print("Predicted Revenue from Budget Model:", budget_prediction)
