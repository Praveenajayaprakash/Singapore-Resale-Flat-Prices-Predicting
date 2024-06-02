import pandas as pd
import numpy as np
import warnings
import streamlit as st
from streamlit_option_menu import option_menu
import pickle

# Load dataset
df = pd.read_csv('cleaned_resale_flat_prices.csv')

# Load the trained model
with open('Regression_Model.pkl', 'rb') as file:
    regg_model = pickle.load(file)

# Mapping Functions
def town_mapping(town_map):
    towns = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG', 
             'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG', 
             'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 
             'PUNGGOL', 'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES', 
             'TOA PAYOH', 'WOODLANDS', 'YISHUN']
    return towns.index(town_map)

def flat_type_mapping(flt_type):
    flat_types = ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION']
    return flat_types.index(flt_type)

def flat_model_mapping(fl_m):
    flat_models = ['2-room', '3Gen', 'Adjoined flat', 'Apartment', 'DBSS', 'Improved', 'Improved-Maisonette', 
                   'Maisonette', 'Model A', 'Model A-Maisonette', 'Model A2', 'Multi Generation', 
                   'New Generation', 'Premium Apartment', 'Premium Apartment Loft', 'Premium Maisonette', 
                   'Simplified', 'Standard', 'Terrace', 'Type S1', 'Type S2']
    return flat_models.index(fl_m)

def predict_price(year, town, flat_type, flr_area_sqm, flat_model, stry_start, stry_end, les_coms_dt, regg_model):
    year_1 = int(year)
    town_2 = town_mapping(town)
    flt_ty_2 = flat_type_mapping(flat_type)
    flr_ar_sqm_1 = int(flr_area_sqm)
    flt_model_2 = flat_model_mapping(flat_model)
    str_str = np.log(int(stry_start) + 1)  # Avoid log(0) by adding 1
    str_end = np.log(int(stry_end) + 1)    # Avoid log(0) by adding 1
    lese_coms_dt = int(les_coms_dt)

    user_data = np.array([[year_1, town_2, flt_ty_2, flr_ar_sqm_1, flt_model_2, str_str, str_end, lese_coms_dt]])
    
    y_pred_1 = regg_model.predict(user_data)
    price = np.exp(y_pred_1[0])

    return round(price)

# Streamlit UI
scrolling_text = "<h1 style='color:red; font-style: italic; font-weight: bold;'><marquee>Welcome to Singapore Flat Resale Project - Your Property Insights Hub</marquee></h1>"
st.markdown(scrolling_text, unsafe_allow_html=True)

def display_home():
    st.markdown(
        """
        <style>
            .big-font {
                font-size: 24px;
                font-weight: bold;
                color: #2f4f4f;
            }
            .info-box {
                background-color: #f0f8ff;
                padding: 15px;
                border-radius: 10px;
                margin: 20px 0;
            }
            .marquee {
                color: #4682b4;
                font-size: 18px;
                white-space: nowrap;
                overflow: hidden;
                border: 1px solid #b0c4de;
                padding: 10px;
                border-radius: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<p class='big-font'>Unlocking Property Insights with Singapore Flat Resale Data</p>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class='info-box'>
            <p><strong>About HDB Resale Flats:</strong></p>
            <p>The Housing & Development Board (HDB) is Singapore's public housing authority. It provides affordable housing to Singaporeans through new and resale flats.</p>
            <p><strong>Uses of Resale Flat Data:</strong></p>
            <ul>
                <li>Understanding Market Trends</li>
                <li>Predicting Future Prices</li>
                <li>Making Informed Buying/Selling Decisions</li>
                <li>Analyzing Factors Affecting Prices</li>
            </ul>
            <p>Explore the Singapore Flat Resale Project to gain valuable insights into the housing market and make informed property decisions.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        The Singapore Flat Resale Project is designed to provide comprehensive insights into the resale flat market in Singapore.
        Dive into the data to discover trends, analyze factors affecting prices, and leverage this information for your property needs.
    """)

def display_about():
    st.title("About the Project")
    scrolling_text = "<h1 style='color:red; font-style: italic; font-weight: bold;'>Welcome to Singapore Flat Resale Project - Your Property Insights Hub</marquee></h1>"
    st.markdown("""
        ### Project Background
        This project aims to provide an accurate prediction of resale prices for HDB flats in Singapore using machine learning techniques.

        ### Methodology
        - **Data Collection:** The data is sourced from the HDB resale flat prices dataset.
        - **Data Preprocessing:** The data is cleaned and preprocessed to ensure it is suitable for modeling.
        - **Modeling:** A linear regression model is used to predict the resale prices based on various features.

        ### Future Work
        - Explore the use of more complex models such as Random Forest or Gradient Boosting for potentially better predictions.
        - Incorporate more features such as nearby amenities, schools, and transportation options to enhance the model.
    """)

with st.sidebar:
    select = option_menu("Main Menu", ["Home", "Price Prediction", "About"])

if select == "Home":
    image_url1="https://miro.medium.com/v2/resize:fit:1400/0*hn4nICHk9Cq-tugt.jpeg"
    st.sidebar.image(image_url1, caption='Singapore Resale - Smart Price', use_column_width=True)
    display_home()   
    
elif select == "Price Prediction":
    image_url1="https://miro.medium.com/v2/resize:fit:1400/0*hn4nICHk9Cq-tugt.jpeg"
    st.sidebar.image(image_url1, caption='Singapore Resale - Smart Price', use_column_width=True)  
    col1, col2 = st.columns(2)
    with col1:
        year = st.selectbox("Select the Year", [str(y) for y in range(2015, 2025)])
        town = st.selectbox("Select the Town", ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
                                                'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
                                                'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
                                                'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
                                                'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
                                                'TOA PAYOH', 'WOODLANDS', 'YISHUN'])
        flat_type = st.selectbox("Select the Flat Type", ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION'])
        flr_area_sqm = st.number_input("Enter the Value of Floor Area sqm (Min: 31 / Max: 280)", min_value=31.0, max_value=280.0)

        flat_model = st.selectbox("Select the Flat Model", ['Improved', 'New Generation', 'Model A', 'Standard', 'Simplified',
                                                            'Premium Apartment', 'Maisonette', 'Apartment', 'Model A2',
                                                            'Type S1', 'Type S2', 'Adjoined flat', 'Terrace', 'DBSS',
                                                            'Model A-Maisonette', 'Premium Maisonette', 'Multi Generation',
                                                            'Premium Apartment Loft', 'Improved-Maisonette', '2-room', '3Gen'])

    with col2:
        stry_start = st.slider("Enter the Value of Storey Start", min_value=1, max_value=40)
        stry_end = st.slider("Enter the Value of Storey End", min_value=1, max_value=40)
        re_les_year = st.slider("Enter the Value of Remaining Lease Year (Min: 42 / Max: 97)", min_value=42, max_value=97)
        re_les_month = st.slider("Enter the Value of Remaining Lease Month (Min: 1 / Max: 11)", min_value=1, max_value=11)
        les_coms_dt = st.selectbox("Select the Lease Commence Date", [str(i) for i in range(1966, 2023)])

    button = st.button("Predict the Price", use_container_width=True)

    if button:
        if stry_start == 0 or stry_end == 0:
            st.write("## :red[Storey Start and Storey End values must be greater than 0.]")
        else:
            pre_price = predict_price(year, town, flat_type, flr_area_sqm, flat_model, stry_start, stry_end, les_coms_dt, regg_model)
            st.write("## :green[**The Predicted Price is:**] ", pre_price)

elif select == "About":
        image_url = "https://miro.medium.com/v2/resize:fit:1400/0*hn4nICHk9Cq-tugt.jpeg"
        st.sidebar.image(image_url, caption='Singapore Resale - Smart Price', use_column_width=True)
        display_about()
