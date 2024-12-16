# predict_model.py

import pandas as pd
import joblib

# Load the trained model
model = joblib.load('trained_model.pkl')

# Define a function to preprocess and predict for new inputs
def predict_crystal_system(model, new_data):
    # Create a DataFrame with the new material formula and features
    new_data_df = pd.DataFrame(new_data)
    
    # Use the trained model to make predictions
    predictions = model.predict(new_data_df)
    
    # Return the predicted crystal systems
    return predictions

# Example of new data to be tested
new_material_data = [
    {
        'Formula': 'LiFePO4', 
        'Space Group Symbol': 'Pna21',
        'Space Group Number': 62, 
        'Sites': 4, 
        'Formation Energy': -0.6, 
        'Volume': 234.5, 
        'Density': 3.6, 
        'Band Gap': 3.2, 
        'Magnetic Ordering': 'None', 
        'Total Magnetization': 0.0
    },
    {
        'Formula': 'NaCoO2', 
        'Space Group Symbol': 'R-3m',
        'Space Group Number': 166, 
        'Sites': 3, 
        'Formation Energy': -0.8, 
        'Volume': 300.4, 
        'Density': 4.5, 
        'Band Gap': 2.8, 
        'Magnetic Ordering': 'None', 
        'Total Magnetization': 0.0
    }
]

# Now call the predict function to get predictions
predicted_crystal_systems = predict_crystal_system(model, new_material_data)

# Output the predictions
for idx, material in enumerate(new_material_data):
    print(f"Formula: {material['Formula']} -> Predicted Crystal System: {predicted_crystal_systems[idx]}")