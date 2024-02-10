from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error 
import pandas as pd 
import shap 
df = pd.read_csv('model/Demographic_Data_Orig.csv')
df.drop(columns=['ip.address', 'full.name'], axis=1, inplace=True)
selected_columns = ['region', 'in.store', 'age', 'items', 'amount']
df = df[selected_columns]
X = df.drop('amount', axis=1) # Features 
y = df['amount'] # Target variable # Split the data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Initialize and train GBM model 
gbm_model = GradientBoostingRegressor(n_estimators=52, max_depth=6, min_samples_split=2, learning_rate=0.1, loss='squared_error') # Corrected loss function name ) 
gbm_model.fit(X_train, y_train) # Make predictions on the test set 
y_pred = gbm_model.predict(X_test) # Evaluate the model 
mse = mean_squared_error(y_test, y_pred) 
print(f'Mean Squared Error on Test Set: {mse}') # Display model summary 
print("Model Summary:") 
print(f"Number of Trees: {gbm_model.n_estimators}") 
print(f"Max Depth: {gbm_model.max_depth}") # Add more model summary information as needed

explainer = shap.TreeExplainer(gbm_model) # Calculate SHAP values 
shap_values = explainer.shap_values(X_test) # Summary plot for feature importance 
shap.summary_plot(shap_values, X_test)


import pickle 
with open('gbm_model.pkl', 'wb') as model_file: 
    pickle.dump(gbm_model, model_file)

with open('shap_explainer.pkl', 'wb') as explainer_file: 
    pickle.dump(explainer, explainer_file)