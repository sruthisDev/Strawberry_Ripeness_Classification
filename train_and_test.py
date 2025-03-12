import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import joblib  # To save and load the trained model

# Step 1: Load the dataset
data = pd.read_csv("fruit_features.csv")  # Replace with actual file

# Step 2: Ignore the Filename column for training but keep it for reference
filenames = data.iloc[:, 0]  # Extract filename column
data = data.iloc[:, 1:]  # Drop the filename column for processing

# Step 3: Split the data into train and test sets
test_data = pd.concat([data.head(8), data.tail(8)])  # Top 8 and bottom 8 rows as test set
train_data = data.iloc[8:-8]  # Excluding test data

test_filenames = pd.concat([filenames.head(8), filenames.tail(8)])  # Extract test filenames
train_filenames = filenames.iloc[8:-8]  # Remaining filenames for training (not needed)

# Step 4: Separate features and labels
X_train = train_data.iloc[:, :-1]  # Features for training
y_train = train_data.iloc[:, -1]   # Labels for training

X_test = test_data.iloc[:, :-1]  # Features for testing
y_test = test_data.iloc[:, -1]   # Actual labels for testing

# Step 5: Normalize the feature data
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)  # Normalize test data using the same scaler

# Step 6: Train the SVM model
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')  # RBF Kernel for non-linearity
svm_model.fit(X_train_normalized, y_train)

# Step 7: Save the model and scaler for future use
joblib.dump(svm_model, "svm_strawberry_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model trained and saved successfully!")

# Step 8: Predict on the test set
y_pred = svm_model.predict(X_test_normalized)

# Step 9: Display results
results = pd.DataFrame({"Filename": test_filenames.values, "Actual": y_test.values, "Predicted": y_pred})
print("\nPrediction Results on Test Data:")
print(results)

# Save results to a CSV file
results.to_csv("strawberry_predictions.csv", index=False)
print("\nPredictions saved to 'strawberry_predictions.csv'")
