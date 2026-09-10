import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import joblib

from model_utils import load_train_test_split

X_train, X_test, y_train, y_test, train_filenames, test_filenames = load_train_test_split()

scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_normalized, y_train)

joblib.dump(knn_model, "knn_strawberry_model.pkl")
joblib.dump(scaler, "knn_scaler.pkl")

print("KNN model trained and saved successfully!")

y_pred = knn_model.predict(X_test_normalized)
accuracy = (y_pred == y_test.values).mean()
print(f"KNN test accuracy: {accuracy:.2%}")

results = pd.DataFrame({"Filename": test_filenames.values, "Actual": y_test.values, "Predicted": y_pred})
print("\nPrediction Results on Test Data:")
print(results)

results.to_csv("knn_predictions.csv", index=False)
print("\nPredictions saved to 'knn_predictions.csv'")
