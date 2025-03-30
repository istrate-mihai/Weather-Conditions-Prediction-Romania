import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib
import streamlit as st

def main():

    # List of Romanian cities (cleaned up Unicode characters)
    cities = [
        'Alba Iulia', 'Arad', 'Pitesti', 'Bacau', 'Oradea', 'Bistrita',
        'Botosani', 'Braila', 'Brasov', 'Bucuresti', 'Buzau',
        'Calarasi', 'Resita', 'Cluj-Napoca', 'Constanta',
        'Sfântu Gheorghe', 'Târgoviste', 'Craiova', 'Galati', 'Giurgiu',
        'Târgu Jiu', 'Miercurea Ciuc', 'Deva', 'Slobozia', 'Iasi', 'Baia Mare',
        'Drobeta-Turnu Severin', 'Târgu Mures', 'Piatra Neamt',
        'Slatina', 'Ploiesti', 'Zalau', 'Satu Mare', 'Sibiu', 'Suceava',
        'Alexandria', 'Timisoara', 'Tulcea', 'Râmnicu Vâlcea', 'Vaslui',
        'Focsani', 'Bihor'
    ]

    # Display the cover image
    st.image('./assets/images/dataset-cover.jpg', use_container_width = True, caption = 'Weather Prediction for Romanian Cities')

    # Title and description
    st.title('Weather Condition Predictor')
    st.write('Enter weather parameters to predict the condition')

    selected_city = st.selectbox(
        'Select City',
        options = sorted(cities),
        index = 0,
    )

    temperature_celsius = st.sidebar.slider('Temperature (°C)',         min_value = -30.0, max_value = 50.0,   value = 20.0)
    relative_humidity   = st.sidebar.slider('Relative Humidity (%)',    min_value = 0.0,   max_value = 100.0,  value = 50.0)
    heat_index_celsius  = st.sidebar.slider('Heat Index (°C)',          min_value = -30.0, max_value = 60.0,   value = 20.0)
    wind_speed          = st.sidebar.slider('Wind Speed (km/h)',        min_value = 0.0,   max_value = 160.0,  value = 15.0)
    precipitation_cover = st.sidebar.slider('Precipitation Cover (%)',  min_value = 0.0,   max_value = 100.0,  value = 0.0)

    snow_depth          = st.sidebar.slider('Snow Depth (cm)',          min_value = 0.0,   max_value = 100.0,  value = 0.0)
    visibility          = st.sidebar.slider('Visibility (km)',          min_value = 0.0,   max_value = 20.0,   value  = 10.0)
    cloud_cover         = st.sidebar.slider('Cloud Cover (%)',          min_value = 0.0,   max_value = 100.0,  value = 0.0)
    sea_level_pressure  = st.sidebar.slider('Sea Level Pressure (hPa)', min_value = 950.0, max_value = 1050.0, value = 1013.0)
    city_label          = cities.index(selected_city)

    weather_conditions = {
        0: 'Partially cloudy',
        1: 'Overcast',
        2: 'Clear',
        3: 'Rain, Overcast',
        4: 'Rain, Partially cloudy',
        5: 'Rain'
    }

    weather_conditions_images = {
        0: './assets/images/partially_cloudly.jpeg',
        1: './assets/images/overcast.jpeg',
        2: './assets/images/clear_weather.jpeg',
        3: './assets/images/rain_overcast.jpeg',
        4: './assets/images/rain_partially_clouded.jpeg',
        5: './assets/images/rain.jpeg',
    }

    if st.button('Predict Weather Condition'):
        # Convert temperature to fahrenheit for the model
        temperature_fahrenheit = (temperature_celsius * 9 / 5) + 32
        heat_index_fahrenheit  = (heat_index_celsius * 9 / 5) + 32
        input_data_for_model   = np.array([[
            temperature_fahrenheit,
            relative_humidity,
            heat_index_fahrenheit,
            wind_speed,
            precipitation_cover,
            snow_depth,
            visibility,
            cloud_cover,
            sea_level_pressure,
            city_label,
        ]])

        try:
            scaler = joblib.load('weather_scaler.joblib')
            model  = joblib.load('weather_knn_model.joblib')

            # Scale the input data using the same scaler
            input_data_scaled = scaler.transform(input_data_for_model)
            input_data        = input_data_for_model[:, :-1]

            # Make prediction
            prediction                = model.predict(input_data_scaled)
            predicted_condition_image = weather_conditions_images[prediction[0]]
            predicted_condition       = weather_conditions[prediction[0]]

            st.write('---')
            st.subheader('Prediction Results')

            st.write(f'**Selected City:** {selected_city}')
            st.write('**Input Parameters:**')

            df = pd.DataFrame(input_data, columns = [
                'Temperature (°F)', 'Relative Humidity (%)', 'Heat Index (°F)', 'Wind Speed (km/h)',
                'Precipitation Cover (%)', 'Snow Depth (cm)', 'Visibility (km)', 'Cloud Cover (%)',
                'Sea Level Pressure (hPa)'
            ])
            print(predicted_condition)

            st.write(df)
            st.markdown(f"""
                <div style='padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
                    <h3 style='color: #1f77b4; margin: 0;'>Predicted Weather Condition: {predicted_condition}</h3>
                </div>""", unsafe_allow_html=True
            )

            # Display the cover image
            st.image(predicted_condition_image, use_container_width = True, caption = 'Predicted Weather Condition')

        except FileNotFoundError:
            st.error('Model or scaler file not found. Make sure to save both the model and scaler first.')

if __name__ == '__main__':
    main()
