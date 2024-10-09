# Advertising Expense and Sales Prediction Model

## Project Introduction

This is an interactive application based on linear regression that predicts sales based on advertising expenses. Users can adjust parameters to see how the model changes.

## Prompt Summary

Request: Provide an example of solving a problem using linear regression, with a requirement for data simulation.

Precise Requirements: The analysis should follow the six steps of the CRISP-DM framework.

Code: The analysis process should be implemented in Python code.

Convert to Streamlit: The code should be transformed into a Streamlit application to enhance interactivity.

Parameter Adjustment: Users should be able to freely adjust the slope, noise, and number of x values.

Interface Adjustment: The parameter adjustment area should be moved above the chart, with parameters displayed in the left column and explanations and charts displayed in the right column.

Visual Effects: Adjust font sizes and make multiple scaling and positioning adjustments.

Organize the Process: Document the entire process as a README for GitHub.

## Technology Stack

- Python
- Streamlit
- Scikit-learn
- NumPy
- Pandas
- Matplotlib

## Installation and Usage

1. Ensure you have a Python environment installed.
2. Install the required libraries:
    ```bash
    pip install streamlit scikit-learn numpy pandas matplotlib
    ```
3. Save the following code as `app.py`.

    ```python
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

    # Create two columns
    col1, col2 = st.columns([1, 3])  # Left column (1/4), right column (3/4)

    # Adjust parameters in the left column
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
    x = np.random.uniform(5, 50, num_points)  # Advertising expense (x)

    # Generate sales data based on user inputs
    noise = np.random.normal(0, noise_level, num_points)  # Random noise
    y = 2 + slope * x + noise  # Sales (y)

    # Organize data into a DataFrame
    data = pd.DataFrame({'Advertising Expense (x)': x, 'Sales (y)': y})

    # Model building
    model = LinearRegression()
    X = data[['Advertising Expense (x)']]
    y = data['Sales (y)']
    model.fit(X, y)

    # Model results
    intercept = model.intercept_
    coef = model.coef_[0]

    # Display model results and visualization in the right column
    with col2:
        st.write(f"### Model Results")
        st.write(f"Intercept: {intercept:.2f}")
        st.write(f"Slope: {coef:.2f} (For each additional thousand dollars in advertising expense, sales increase by about {coef:.2f} thousand dollars)")

        # Model evaluation
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        st.write(f"Model R-squared: {r2:.2f} (Model Fit Quality)")

        # Data and model visualization
        fig, ax = plt.subplots(figsize=(6, 4))  # Set figure size
        ax.scatter(x, y, color='blue', label='Actual Data', s=5)
        ax.plot(x, y_pred, color='red', label='Fitted Line')
        ax.set_xlabel('Advertising Expense (in thousand dollars)', fontsize=4)
        ax.set_ylabel('Sales (in thousand dollars)', fontsize=4)
        ax.set_title('Linear Regression: Advertising Expense vs Sales', fontsize=5)
        ax.legend(fontsize=3)
        ax.tick_params(axis='both', which='major', labelsize=3)
        ax.tick_params(axis='both', which='minor', labelsize=2)
        ax.grid(True)

        # Embed the chart in Streamlit
        st.pyplot(fig)

    # Expander for additional information
    with st.expander("See Additional Information", expanded=False):
        st.write("Adjust the parameters above to see how the model changes.")
    ```

4. Run the application in the command line:
    ```bash
    streamlit run app.py
    ```

5. Demonstration
   
![Demo_Image](https://github.com/yao790609/AIoT-DA_HW1/blob/main/AIoT-DA_2024_HW1.png)
## Contribution

If you have any questions or suggestions, feel free to raise them.
