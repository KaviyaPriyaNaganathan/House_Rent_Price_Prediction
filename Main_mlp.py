#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("G:\MLP\House_Rent_Dataset.csv")

# Preprocess the data
data["Area Type"] = data["Area Type"].map({"Super Area": 1, "Carpet Area": 2, "Built Area": 3})
data["City"] = data["City"].map({"Mumbai": 4000, "Chennai": 6000, 
                                 "Bangalore": 5600, "Hyderabad": 5000, 
                                 "Delhi": 1100, "Kolkata": 7000})
data["Furnishing Status"] = data["Furnishing Status"].map({"Unfurnished": 0, 
                                                           "Semi-Furnished": 1, 
                                                           "Furnished": 2})
data["Tenant Preferred"] = data["Tenant Preferred"].map({"Bachelors/Family": 2, 
                                                         "Bachelors": 1, 
                                                         "Family": 3})

# Splitting data
x = np.array(data[["BHK", "Size", "Area Type", "City", 
                   "Furnishing Status", "Tenant Preferred", 
                   "Bathroom"]])
y = np.array(data[["Rent"]])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)

# Define and train the model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(xtrain.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(xtrain, ytrain, batch_size=1, epochs=21)

# Create a Tkinter window
window = tk.Tk()
window.title("House Rent Prediction")

# Function to predict rent
def predict_rent():
    try:
        # Get inputs from the user
        bhk = int(entry_bhk.get())
        size = int(entry_size.get())
        area_type = int(entry_area_type.get())
        city = int(entry_city.get())
        furnishing_status = int(entry_furnishing_status.get())
        tenant_type = int(entry_tenant_type.get())
        bathrooms = int(entry_bathrooms.get())
        
        # Predict rent using the model
        features = np.array([[bhk, size, area_type, city, furnishing_status, tenant_type, bathrooms]])
        predicted_rent = model.predict(features)
        
        # Display the predicted rent
        messagebox.showinfo("Prediction", f"Predicted House Rent = {predicted_rent[0][0]}")
        
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numerical values.")

# Create input fields
tk.Label(window, text="Number of BHK:").grid(row=0, column=0)
entry_bhk = tk.Entry(window)
entry_bhk.grid(row=0, column=1)

tk.Label(window, text="Size of the House:").grid(row=1, column=0)
entry_size = tk.Entry(window)
entry_size.grid(row=1, column=1)

tk.Label(window, text="Area Type (1: Super Area, 2: Carpet Area, 3: Built Area):").grid(row=2, column=0)
entry_area_type = tk.Entry(window)
entry_area_type.grid(row=2, column=1)

tk.Label(window, text="Pin Code of the City:").grid(row=3, column=0)
entry_city = tk.Entry(window)
entry_city.grid(row=3, column=1)

tk.Label(window, text="Furnishing Status (0: Unfurnished, 1: Semi-Furnished, 2: Furnished):").grid(row=4, column=0)
entry_furnishing_status = tk.Entry(window)
entry_furnishing_status.grid(row=4, column=1)

tk.Label(window, text="Tenant Type (1: Bachelors, 2: Bachelors/Family, 3: Only Family):").grid(row=5, column=0)
entry_tenant_type = tk.Entry(window)
entry_tenant_type.grid(row=5, column=1)

tk.Label(window, text="Number of Bathrooms:").grid(row=6, column=0)
entry_bathrooms = tk.Entry(window)
entry_bathrooms.grid(row=6, column=1)

# Create predict button
predict_button = tk.Button(window, text="Predict Rent", command=predict_rent)
predict_button.grid(row=7, columnspan=2)

# Run the Tkinter event loop
window.mainloop()

