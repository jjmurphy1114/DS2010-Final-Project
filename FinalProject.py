import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('CleanedMovieSet.csv')

# Convert revenue and budget columns to numeric (remove commas and dollar signs)
df['revenue'] = df['revenue'].replace('[\$,]', '', regex=True).astype(float)
df['budget'] = df['budget'].replace('[\$,]', '', regex=True).astype(float)

# Display the first few rows
print(df.head())

# Preprocessing: Extract necessary columns
df['genres'] = df['genres'].str.split(', ').str[0]  # Use the first genre listed
df = df.rename(columns={
    'vote_average': 'AudienceRating',
    'vote_count': 'numVotes',
    'budget': 'Budget',
    'revenue': 'Revenue',
    'averageRating': 'CriticRating'
})

# Conjecture 1: Genre Impact on Box Office Revenue
# One-hot encode the Genre column
genre_dummies = pd.get_dummies(df['genres'], drop_first=True)

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

# Conjecture 2: Ratings (Audience and Critical) Correlation with Revenue
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

# Conjecture 3: Budget Impact on Revenue with Diminishing Returns
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

# Sort data by budget to ensure a smooth line
df_sorted = df.sort_values(by='Budget')

# Generate predictions for the sorted data
predicted_revenue = model_budget.predict(df_sorted[['Budget', 'BudgetSquared']])

# Plot the actual vs predicted revenue
plt.figure(figsize=(10, 6))
plt.scatter(df['Budget'], df['Revenue'], label='Actual Data', alpha=0.6)
plt.plot(df_sorted['Budget'], predicted_revenue, color='red', label='Model Prediction')
plt.xlabel('Budget (Millions $)')
plt.ylabel('Revenue (Millions $)')
plt.legend()
plt.title('Budget vs Revenue with Predicted Line')
plt.show()

# Predictions for New Data
# Example new data for predictions
new_data = {
    'genres': ['Action', 'Horror'],  # First genre in the list
    'AudienceRating': [7.5, 6.8],
    'CriticRating': [80, 65],
    'Budget': [50_000_000, 10_000_000]
}

new_df = pd.DataFrame(new_data)

# One-hot encode genres for predictions
genre_features = pd.get_dummies(new_df['genres'], drop_first=True)
genre_features = genre_features.reindex(columns=genre_dummies.columns, fill_value=0)

# Add quadratic budget term
new_df['BudgetSquared'] = new_df['Budget'] ** 2

# Predictions
genre_prediction = model_genre.predict(genre_features)
ratings_prediction = model_ratings.predict(new_df[['AudienceRating', 'CriticRating']])
budget_prediction = model_budget.predict(new_df[['Budget', 'BudgetSquared']])

print("Predicted Revenue from Genre Model:", genre_prediction)
print("Predicted Revenue from Ratings Model:", ratings_prediction)
print("Predicted Revenue from Budget Model:", budget_prediction)
