import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

RANDOM_STATE = 42

# Step 1: Load the dataset
data = pd.read_csv("fruit_features.csv")

# Step 2: Separate filenames, features, and labels
filenames = data["Filename"]
X = data.drop(columns=["Filename", "Label"])
y = data["Label"]

# Step 3: Randomized, stratified train/test split
X_train, X_test, y_train, y_test, train_filenames, test_filenames = train_test_split(
    X, y, filenames, test_size=0.15, stratify=y, random_state=RANDOM_STATE
)

# Step 4: Normalize the feature data
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)  # Normalize test data using the same scaler

# Step 5: Train the SVM model
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)  # RBF Kernel for non-linearity
svm_model.fit(X_train_normalized, y_train)

# Step 6: Save the model and scaler for future use
joblib.dump(svm_model, "svm_strawberry_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model trained and saved successfully!")

# Step 7: Predict on the test set
y_pred = svm_model.predict(X_test_normalized)

# Step 8: Display results
results = pd.DataFrame({"Filename": test_filenames.values, "Actual": y_test.values, "Predicted": y_pred})
print("\nPrediction Results on Test Data:")
print(results)

# Save results to a CSV file
results.to_csv("strawberry_predictions.csv", index=False)
print("\nPredictions saved to 'strawberry_predictions.csv'")
