import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data from the Excel file
data = pd.read_excel('C:/Users/DaughertTL18/Downloads/baseball.xlsx')

# Select the features (RS, RA, W, OBP, SLG, BA) and the target variable (Playoffs)
X = data[['Runs Scored', 'Runs Allowed', 'Wins', 'OBP', 'SLG', 'Team Batting Average']]
y = data['Playoffs']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Print the predictions
print("Predictions:", y_pred)
print("go sox")
