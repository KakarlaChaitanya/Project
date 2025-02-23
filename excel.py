import pandas as pd
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  

# 1. Read data from Excel
data = pd.read_excel("D:\New folder")  # Replace with your file path
X = data.drop("target_column", axis=1)  # Replace with your target column name
y = data["target_column"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
new_data = ...  
predictions = model.predict(new_data)
predictions_df = pd.DataFrame({"predicted_value": predictions})
predictions_df.to_excel("predictions.xlsx", index=False)

print("Predictions generated!")
