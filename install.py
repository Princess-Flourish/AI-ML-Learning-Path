install numpy as np
install pickle

# Create a function for prediction
def california_house_price_prediction(input_data):
    # Convert input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the input data to match the shape expected by the model
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make a prediction using the loaded model
    prediction = loaded_model.predict(input_data_reshaped)

    # Print the prediction
    print(prediction)


def main():
    # Set the title of the web app
    st.title('California House Price Prediction Web App')
    
    # Text inputs for various features
    longitude = st.text_input('Longitude of Location')
    latitude = st.text_input('Latitude of Location')
    housing_median_age = st.text_input('Median Age House Value')
    total_rooms = st.text_input('House Total Room Value')
    total_bedrooms = st.text_input('House Total Bedroom Value')
    population = st.text_input('Population Value')
    households = st.text_input('Household Value')
    median_income = st.text_input('Median Income Value')
    ocean_proximity = st.text_input('Ocean Proximity of House')

    # Initialize variable to store predicted house price
    house_price = ''

    # If the button is clicked, predict the house price
    if st.button('House Price Prediction'):
        # Call the function to predict house price
        house_price = california_house_price_prediction([longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity])
    
    # Display the predicted house price
    st.success(house_price)



if __name__ == '__main__':
    main()