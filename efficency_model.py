import pandas as pd
import joblib

# Load the trained model
model = joblib.load('trained_model.pkl')

# Load the dataset (use the same dataset used for training)
df = pd.read_csv("training_dataset.csv")

# Selecting features
X = df[['Formula', 'Crystal System', 'Space Group Symbol', 'Space Group Number', 
        'Sites', 'Formation Energy', 'Volume', 'Density', 'Band Gap', 'Magnetic Ordering', 
        'Total Magnetization']]

# Predict the crystal systems using the trained model
predictions = model.predict(X)

# Add predicted crystal system to the original DataFrame
df['Predicted Crystal System'] = predictions

# Now, let's analyze the features associated with each crystal system
# Focus on efficiency-related features: Density, Formation Energy, Band Gap
efficiency_data = df[['Crystal System', 'Density', 'Formation Energy', 'Band Gap', 'Predicted Crystal System']]

# Group by the predicted crystal system and calculate the mean of the efficiency features
# Only apply the mean to the numeric columns
numeric_columns = ['Density', 'Formation Energy', 'Band Gap']
grouped_efficiency = efficiency_data.groupby('Predicted Crystal System')[numeric_columns].mean()

# Sort by the efficiency-related features to find which crystal system is most efficient
# For example, prioritize high Density, low Formation Energy, and low Band Gap
sorted_efficiency = grouped_efficiency.sort_values(by=['Density', 'Formation Energy', 'Band Gap'], ascending=[False, True, True])

print("Crystal System Efficiency Analysis:")
print(sorted_efficiency)