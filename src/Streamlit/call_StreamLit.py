import streamlit as st
import pandas as pd

# Title of the app


# Streamlit App
def call_ST(model, test_data, test_true):
    st.title("Anomaly Detection")

        # Make predictions
    if st.button("Run Model"):
        results = model.predict(test_data)
        st.write("Predictions")
        st.write(results)
