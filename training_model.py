import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from preprocessing import df

# Encode categorical variable
label_encoder = LabelEncoder()
df['Composition'] = label_encoder.fit_transform(df['Composition'])

# Step 3: Split data into features and target
X = df.drop(columns=['class'])
y = df['class']

# Step 4: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Step 7: Make predictions
y_pred = rf_model.predict(X_test_scaled)

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 9: Hyperparameter Tuning (optional)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), 
                           param_grid=param_grid, 
                           cv=5, 
                           n_jobs=-1, 
                           verbose=2)
grid_search.fit(X_train_scaled, y_train)

best_rf_model = grid_search.best_estimator_
y_pred_best = best_rf_model.predict(X_test_scaled)
best_accuracy = accuracy_score(y_test, y_pred_best)
print(f'Best Model Accuracy: {best_accuracy:.4f}')

# Step 10: Save the model
joblib.dump(best_rf_model, 'battery_efficiency_model.pkl')

import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix
corr_matrix = df.corr()

sns.regplot(x='avg_Crystal radius (Å)', y='Eg', data=df, scatter_kws={'alpha': 0.7}, line_kws={'color': 'red'})
plt.title("Crystal Radius vs. Band Gap with Regression Line")
plt.xlabel("Crystal Radius (Å)")
plt.ylabel("Band Gap (Eg)")
plt.show()