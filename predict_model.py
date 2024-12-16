import pandas as pd
import joblib

# Load the trained model
model = joblib.load('trained_model.pkl')

# Define a function to preprocess and predict for new inputs
def predict_crystal_system(model, input_file):
    # Read the CSV file into a DataFrame
    new_data_df = pd.read_csv(input_file)
    
    # Ensure the DataFrame has the same columns as the model expects
    expected_columns = ['Formula', 'Space Group Symbol', 'Space Group Number', 'Sites',
                        'Formation Energy', 'Volume', 'Density', 'Band Gap',
                        'Magnetic Ordering', 'Total Magnetization']
    if not all(col in new_data_df.columns for col in expected_columns):
        missing_cols = set(expected_columns) - set(new_data_df.columns)
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Use the trained model to make predictions
    predictions = model.predict(new_data_df)
    
    # Return the DataFrame with predictions
    new_data_df['Predicted Crystal System'] = predictions
    return new_data_df

# Input CSV file containing new material data
input_file = 'input_dataset.csv'

# Call the predict function
predicted_results = predict_crystal_system(model, input_file)

# Output the predictions
print(predicted_results[['Formula', 'Predicted Crystal System']])