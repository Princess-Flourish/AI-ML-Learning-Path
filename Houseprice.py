import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
df = pd.read_csv('California_House_Price.csv')

# Load the trained model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Create a one-hot encoder for the ocean proximity column
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(df[['ocean_proximity']])

# Function for house price prediction
def house_price_prediction(input_data):
    # Extract the ocean proximity value
    ocean_proximity = input_data[-1]
    input_data = input_data[:-1]

    # Convert numerical input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=np.float64).reshape(1, -1)

    # One-hot encode the ocean proximity value
    ocean_proximity_encoded = encoder.transform([[ocean_proximity]])

    # Combine the numerical input data with the encoded ocean proximity
    input_data_final = np.hstack((input_data_as_numpy_array, ocean_proximity_encoded))

    # Make a prediction using the loaded model
    prediction = loaded_model.predict(input_data_final)

    # Return the prediction
    return prediction

# Function to display scatter plot
def display_scatter_plot():
    # Create a new figure with a specified size (10 inches wide by 6 inches tall)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a scatter plot using seaborn
    sns.scatterplot(x='total_rooms', y='population', data=df, ax=ax)

    # Set the label for the x-axis
    ax.set_xlabel('Total Rooms')

    # Set the label for the y-axis
    ax.set_ylabel('Population')

    # Set the title for the plot
    ax.set_title('Total Rooms vs Population')

    # Display the plot
    st.pyplot(fig)

# Function to display joint plot
def display_joint_plot():
    # Create a joint plot (scatter plot with histograms) using seaborn
    fig = sns.jointplot(x=df.latitude.values, y=df.longitude.values, height=10)

    # Set the label for the y-axis (longitude) with a font size of 12
    fig.set_axis_labels('latitude', 'longitude', fontsize=12)

    # Display the plot
    st.pyplot(fig)

# Function to display bar plot
def display_bar_plot():
    # Calculate the average median_house_value for each ocean proximity category
    avg_median_house_value_per_proximity = df.groupby('ocean_proximity')['median_house_value'].mean().reset_index()

    # Create a bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='ocean_proximity', y='median_house_value', data=avg_median_house_value_per_proximity, palette='viridis', ax=ax)

    # Add labels and title
    ax.set_xlabel('Ocean Proximity', fontsize=12)
    ax.set_ylabel('Average house_value', fontsize=12)
    ax.set_title('Average Housing house_value by Ocean Proximity', fontsize=15)

    # Display the plot
    st.pyplot(fig)

# Main function
def main():
    # Set the title of the web app
    st.title('CALIFORNIA HOUSE PRICE PREDICTION WEB APP')

    # Create a dropdown to select page
    page = st.selectbox("Select Page", ["House Price Prediction", "Data Visualization"])

    # Display selected page
    if page == "House Price Prediction":
        # Header Image
        st.image('AI-ML-Learning-Path/houseimage5.jpeg', use_column_width=True)

        # Custom CSS for background color
        st.markdown(
            """
            <style>
            body {
                background-color: #f0f2f6;  /* Set background color to light gray */
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Sidebar for Input
        st.sidebar.title('ENTER HOUSE DETAILS')
        st.sidebar.image('AI-ML-Learning-Path/houseimage2.jpeg', width=150, use_column_width=True)
        longitude = st.sidebar.text_input('Longitude of Location')
        latitude = st.sidebar.text_input('Latitude of Location')
        housing_median_age = st.sidebar.text_input('Median Age House Value')
        total_rooms = st.sidebar.text_input('House Total Room Value')
        total_bedrooms = st.sidebar.text_input('House Total Bedroom Value')
        population = st.sidebar.text_input('Population Value')
        households = st.sidebar.text_input('Household Value')
        median_income = st.sidebar.text_input('Median Income Value')
        ocean_proximity = st.sidebar.selectbox('Ocean Proximity', df['ocean_proximity'].unique())

        # Initialize variable to store predicted house price
        house_price = ''

        # Create columns for layout
        col1, col2, col3 = st.columns([1, 1, 1])

        with col3:
            # If the button is clicked, predict the house price
            if st.button('Predict House Price'):
                # Call the function to predict house price
                input_data = [longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity]
                try:
                    input_data = list(map(float, input_data[:-1])) + [input_data[-1]]
                    house_price = house_price_prediction(input_data)
                except ValueError:
                    st.error("Please ensure all numerical values are entered correctly.")

        # Display the predicted house price
        if house_price:
            st.success(f'Predicted House Price: ${house_price[0]:,.2f}')
    elif page == "Data Visualization":
        # Create a dropdown to select chart
        chart_type = st.selectbox("Select Chart", ["Scatter Plot", "Joint Plot", "Bar Plot"])

        # Display selected chart
        if chart_type == "Scatter Plot":
            display_scatter_plot()
        elif chart_type == "Joint Plot":
            display_joint_plot()
        elif chart_type == "Bar Plot":
            display_bar_plot()

if __name__ == '__main__':
    main()
