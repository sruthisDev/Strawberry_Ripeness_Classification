import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import MinMaxScaler
import joblib

from model_utils import load_train_test_split

X_train, X_test, y_train, y_test, train_filenames, test_filenames = load_train_test_split()

scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
knn_model = KNeighborsClassifier(n_neighbors=5)

ensemble_model = VotingClassifier(estimators=[('svm', svm_model), ('knn', knn_model)], voting='soft')
ensemble_model.fit(X_train_normalized, y_train)

joblib.dump(ensemble_model, "ensemble_strawberry_model.pkl")
joblib.dump(scaler, "ensemble_scaler.pkl")

print("Ensemble model trained and saved successfully!")

y_pred = ensemble_model.predict(X_test_normalized)
accuracy = (y_pred == y_test.values).mean()
print(f"Ensemble (SVM + KNN) test accuracy: {accuracy:.2%}")

results = pd.DataFrame({"Filename": test_filenames.values, "Actual": y_test.values, "Predicted": y_pred})
print("\nPrediction Results on Test Data:")
print(results)

results.to_csv("ensemble_predictions.csv", index=False)
print("\nPredictions saved to 'ensemble_predictions.csv'")
