import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
# Use joblib consistently for sklearn models
from joblib import dump


# Load the dataset
df = pd.read_csv('C:\\Users\\Renugapalraj\\OneDrive\\Desktop\\FINAL YR PRO\\Expanded_Crop_Recommendation.csv')

# Features and labels
X = df.drop('label', axis=1)
y = df['label']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"✅ Model Accuracy on Test Set: {accuracy * 100:.2f}%\n")
#print("📊 Classification Report:")
#print(report)
# Save the trained model (BEST PRACTICE FOR LARGE NUMPY MODELS)
dump(clf, 'crop_recommendation_model.joblib')

print("📁 Model saved to: crop_recommendation_model.joblib")
import pickle
with open('crop_recommendation_model.pkl', 'rb') as f:
    content = pickle.load(f)
print(type(content)) 
