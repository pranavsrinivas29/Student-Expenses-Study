# app.py

import streamlit as st
import pandas as pd
import numpy as np
from model import load_data, preprocess_data, train_model, save_model, load_model
import matplotlib.pyplot as plt


# Streamlit app
def main():
    st.title("Student Expenses Prediction App")

    # Load data
    data = load_data()

    # Display data
    if st.checkbox("Show Data"):
        st.write(data.head())

    # Train model
    if st.button("Train Model"):
        X_train, X_test, y_train, y_test, label_encoders = preprocess_data(data)
        model = train_model(X_train, y_train)
        save_model(model, label_encoders)
        st.success("Model trained and saved!")

    # Load model
    model, label_encoders = load_model()

    # Make prediction
    st.header("Make a Prediction")
    input_data = {}
    for col in data.columns:
        if col not in ['Unnamed: 0', 'monthly_income']:
            input_data[col] = st.text_input(col)

    # Check for missing input fields
    if st.button("Predict"):
        missing_values = [col for col, val in input_data.items() if val == '']
        if missing_values:
            st.error(f"Please fill in all fields: {', '.join(missing_values)}")
        else:
            input_df = pd.DataFrame([input_data])

            # Encode categorical variables
            for column in input_df.columns:
                if column in label_encoders:
                    try:
                        input_df[column] = label_encoders[column].transform(input_df[column])
                    except ValueError as e:
                        st.error(f"Error encoding {column}: {e}")
                        return

            try:
                prediction = model.predict(input_df)
                st.write(f"Predicted Monthly Income: {prediction[0]}")
            except Exception as e:
                st.error(f"Error making prediction: {e}")

# Plotting example using Matplotlib
    st.header("Expense Distribution Plot")
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    x_axis = st.selectbox("Choose X-axis:", numeric_columns)
    y_axis = st.selectbox("Choose Y-axis:", numeric_columns)

    if st.button("Show Plot"):
        if x_axis and y_axis:
            fig, ax = plt.subplots()
            ax.scatter(data[x_axis], data[y_axis])
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            ax.set_title(f'{x_axis} vs {y_axis}')
            st.pyplot(fig)
        else:
            st.error("Please select both X and Y axes.")


if __name__ == '__main__':
    main()
