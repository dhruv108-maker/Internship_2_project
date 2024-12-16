import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
df = pd.read_csv("training_dataset.csv")

# Selecting features and target variable
X = df[['Formula', 'Crystal System', 'Space Group Symbol', 'Space Group Number', 
        'Sites', 'Formation Energy', 'Volume', 'Density', 'Band Gap', 'Magnetic Ordering', 
        'Total Magnetization']]

# Target variable is categorical
y = df['Crystal System']  

# Preprocessing pipeline
numeric_features = ['Space Group Number', 'Sites', 'Formation Energy', 'Volume', 'Density', 'Band Gap', 'Total Magnetization']
categorical_features = ['Formula', 'Space Group Symbol', 'Magnetic Ordering']

# Imputing missing numeric values with median
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with preprocessor and model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model to a file
joblib.dump(model, 'trained_model.pkl') 