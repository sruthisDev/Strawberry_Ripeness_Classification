import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import joblib

from model_utils import load_train_test_split

# Step 1-3: Load the dataset and split into a randomized, stratified train/test set
X_train, X_test, y_train, y_test, train_filenames, test_filenames = load_train_test_split()

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
accuracy = (y_pred == y_test.values).mean()
print(f"SVM test accuracy: {accuracy:.2%}")

# Step 8: Display results
results = pd.DataFrame({"Filename": test_filenames.values, "Actual": y_test.values, "Predicted": y_pred})
print("\nPrediction Results on Test Data:")
print(results)

# Save results to a CSV file
results.to_csv("strawberry_predictions.csv", index=False)
print("\nPredictions saved to 'strawberry_predictions.csv'")
