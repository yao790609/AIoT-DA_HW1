import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import streamlit as st
import matplotlib.pyplot as plt

# Set page configuration to wide mode
st.set_page_config(layout="wide")

# Title and introduction
st.title("Advertising Expense and Sales Prediction Model")
st.write("This is an interactive application based on linear regression that predicts sales based on advertising expenses.")

# Create two columns for layout
col1, col2 = st.columns([1, 3])  # Left column (1/4), right column (3/4)

# Empty space to center the parameters section vertically
with col1:
    st.write("")  # Add empty space to position the Adjust Parameters section
    st.write("")  # Add another line to increase space
    st.write("")  # Additional space
    st.write("")  # Additional space
    st.write("")  # Additional space
    st.write("")  # Additional space
    st.write("")  # Additional space
    st.write("")  # Additional space
    st.write("")  # Additional space
    st.write("")  # Additional space
    st.write("")  # Additional space
    st.write("")  # Additional space
    st.write("")  # Additional space
    st.write("")  # Additional space
    st.write("### Adjust Parameters")
    slope = st.slider("Adjust Slope (Range 0~100):", min_value=0.0, max_value=100.0, value=4.5)
    noise_level = st.slider("Adjust Noise Level (Range 0~100):", min_value=0.0, max_value=100.0, value=8.0)
    num_points = st.slider("Adjust Number of x Values (Range 10~500):", min_value=10, max_value=500, value=30)

    # Simulate generating advertising expense and sales data
    np.random.seed(42)
    x = np.random.uniform(5, 50, num_points)  # Advertising expense (x), range from 5 to 50 thousand dollars

    # Generate sales data based on user inputs
    noise = np.random.normal(0, noise_level, num_points)  # Random noise
    y = 2 + slope * x + noise  # Sales (y) generated based on the formula

    # Organize data into DataFrame
    data = pd.DataFrame({'Advertising Expense (x)': x, 'Sales (y)': y})

    # Model building
    model = LinearRegression()
    X = data[['Advertising Expense (x)']]
    y = data['Sales (y)']
    model.fit(X, y)

    # Model results
    intercept = model.intercept_
    coef = model.coef_[0]

# Right column for model results and visualization
with col2:
    st.write(f"### Model Results")
    st.write(f"Intercept: {intercept:.2f}")
    st.write(f"Slope: {coef:.2f} (For each additional thousand dollars in advertising expense, sales increase by about {coef:.2f} thousand dollars)")

    # Evaluation
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    st.write(f"Model R-squared: {r2:.2f} (Model Fit Quality)")

    # Data and model visualization
    fig, ax = plt.subplots(figsize=(6, 4))  # Set figure size to half of original size
    ax.scatter(x, y, color='blue', label='Actual Data', s=5)  # Adjust marker size
    ax.plot(x, y_pred, color='red', label='Fitted Line')
    
    # Set labels and title with much smaller font size (1/4 of original)
    ax.set_xlabel('Advertising Expense (in thousand dollars)', fontsize=4)  # Original size / 4
    ax.set_ylabel('Sales (in thousand dollars)', fontsize=4)  # Original size / 4
    ax.set_title('Linear Regression: Advertising Expense vs Sales', fontsize=5)  # Original size / 4
    ax.legend(fontsize=3)  # Original size / 4
    
    # Adjust tick label sizes to be 1/4 of original
    ax.tick_params(axis='both', which='major', labelsize=3)  # Original size / 4
    ax.tick_params(axis='both', which='minor', labelsize=2)  # Original size / 4

    ax.grid(True)

    # Embed the chart in Streamlit
    st.pyplot(fig)

# Use expander for additional information if needed
with st.expander("See Additional Information", expanded=False):
    st.write("Adjust the parameters above to see how the model changes.")
