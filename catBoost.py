import catboost
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load your dataset from a CSV file
#file_path = 'strategyResultsG-long.csv'
#modelName = 'predictionG-long.cbm'
file_path = 'strategyResultsA-long.csv'
modelName = 'prediction-long.cbm'
#file_path = 'strategyResultsA-short.csv'
#modelName = 'prediction-short.cbm'
#file_path = 'strategyResultsG-short.csv'
#modelName = 'predictionG-short.cbm'

data = pd.read_csv(file_path)

# Assuming you have a column 'outcome' that indicates the outcome (target variable)
X = data.drop('outcome', axis=1)
y = data['outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a CatBoost classifier
model = CatBoostClassifier(iterations=500, depth=10, learning_rate=0.05, loss_function='Logloss')

# Convert the training data to CatBoost Pool format
categorical_features = ['M5','M10','M15','M30','H1','H2','H4','D1']
train_pool = Pool(data=X_train, label=y_train, cat_features=categorical_features)

# Train the model
model.fit(train_pool, eval_set=(X_test, y_test), verbose=100)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Save the trained model to a file
model.save_model(modelName)

# Load the model from a file (for demonstration purposes)
loaded_model = CatBoostClassifier()
loaded_model.load_model(modelName)

# Make predictions using the loaded model
#new_data = pd.DataFrame({"feature1": [value1], "feature2": [value2], ...})  # Replace with your actual data
#new_predictions = loaded_model.predict(new_data)
#print(f"New predictions: {new_predictions}")