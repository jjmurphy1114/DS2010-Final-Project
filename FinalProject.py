import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('CleanedMovieSet.csv')

# # Convert revenue and budget columns to numeric (remove commas and dollar signs)
df['revenue'] = df['revenue'].replace('[\$,]', '', regex=True).astype(float)
df['budget'] = df['budget'].replace('[\$,]', '', regex=True).astype(float)

# Preprocessing: Extract necessary columns
df['genres'] = df['genres'].str.split(', ').str[0]  # Use the first genre listed
df = df.rename(columns={
    'vote_count': 'numVotes',
    'budget': 'Budget',
    'revenue': 'Revenue',
})

# Conjecture 1: Genre Impact on Box Office Revenue
def graph_conjecture_1():
    # Remove rows with missing or invalid runtime and revenue values
    runtime_data = df[['runtime', 'Revenue']].dropna()
    runtime_data = runtime_data[runtime_data['Revenue'] > 0]

    # Prepare features (runtime) and target (revenue)
    X_runtime = runtime_data[['runtime']]
    y_revenue = runtime_data['Revenue']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_runtime, y_revenue, test_size=0.2, random_state=42)

    # Train the linear regression model
    model_runtime = LinearRegression()
    model_runtime.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model_runtime.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print("Conjecture 1 (Runtime): R2 Score =", r2)

    # Plot the actual vs predicted data
    plt.figure(figsize=(10, 6))
    plt.scatter(X_runtime, y_revenue, alpha=0.6, label="Actual Data")
    plt.plot(X_runtime, model_runtime.predict(X_runtime), color='red', label="Best Fit Line")
    plt.title('Movie Runtime vs Box Office Revenue')
    plt.xlabel('Runtime (minutes)')
    plt.ylabel('Revenue (Millions $)')
    plt.legend()
    plt.show()




# Conjecture 2: Audience Rating Correlation with Revenue
def graph_conjecture_2():
    # Remove rows with missing or invalid Rating and Revenue values
    ratings_data = df[['averageRating', 'Revenue']].dropna()  # Include Revenue in the filtered data
    ratings_data = ratings_data[ratings_data['Revenue'] > 0]  # Ensure only positive Revenue values

    # Prepare features (Rating) and target (Revenue)
    X_ratings = ratings_data[['averageRating']]
    y_revenue = ratings_data['Revenue']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_ratings, y_revenue, test_size=0.2, random_state=42)

    # Train the linear regression model
    model_ratings = LinearRegression()
    model_ratings.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model_ratings.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print("Conjecture 2 (Rating): R2 Score =", r2)

    # Plot the actual vs predicted revenue for averageRating
    plt.figure(figsize=(10, 6))
    plt.scatter(ratings_data['averageRating'], ratings_data['Revenue'], alpha=0.6, label="Actual Data")
    plt.plot(
        ratings_data['averageRating'],
        model_ratings.predict(ratings_data[['averageRating']]),
        color='red',
        label="Prediction Line"
    )
    plt.title('Rating vs Revenue')
    plt.xlabel('Rating')
    plt.ylabel('Revenue (Millions $)')
    plt.legend()
    plt.show()


# Conjecture 3: Budget Impact on Revenue with Diminishing Returns
def graph_conjecture_3():
    df['BudgetSquared'] = df['Budget'] ** 2
    X_budget = df[['Budget', 'BudgetSquared']]
    y_revenue = df['Revenue']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_budget, y_revenue, test_size=0.2, random_state=42)

    # Train model
    model_budget = LinearRegression()
    model_budget.fit(X_train, y_train)

    # Evaluate model
    y_pred = model_budget.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print("Conjecture 3 (Budget): R2 Score =", r2)

    # Sort data for a smooth prediction line
    df_sorted = df.sort_values(by='Budget')
    predicted_revenue = model_budget.predict(df_sorted[['Budget', 'BudgetSquared']])

    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Budget'], df['Revenue'], alpha=0.6, label='Actual Data')
    plt.plot(df_sorted['Budget'], predicted_revenue, color='red', label='Prediction')
    plt.title('Budget vs Revenue with Predicted Line')
    plt.xlabel('Budget')
    plt.ylabel('Revenue')
    plt.legend()
    plt.show()

# Prompt user to select a conjecture to visualize
while True:
    user_input = input("Enter 1 to visualize Genre impact, 2 for  Ratings impact, or 3 for Budget impact (or 'exit' to quit): ")
    if user_input == '1':
        graph_conjecture_1()
    elif user_input == '2':
        graph_conjecture_2()
    elif user_input == '3':
        graph_conjecture_3()
    elif user_input.lower() == 'exit':
        print("Exiting...")
        break
    else:
        print("Invalid input. Please enter 1, 2, 3, or 'exit'.")
