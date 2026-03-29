import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBRegressor
import base64

def add_bg_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

add_bg_local("D:\Predictive analysis\Project\Cricket_stadium.jpg")

label_style = """
<style>
/* Label text above all input boxes */
.stSelectbox label, .stNumberInput label, .stTextInput label {
    font-size: 50px ;      /* Change size */
    color: #ffffff !important;       /* White color */
    font-weight: 700 !important;     /* Make bold */
    text-shadow: 1px 1px 4px #000;   /* Add glow for visibility */
}
</style>
"""
st.markdown(label_style, unsafe_allow_html=True)

pipe_xgb = pickle.load(open('D:\Predictive analysis\Project\project codes\pipe.pkl','rb'))
pipe_lgbm = pickle.load(open('D:\Predictive analysis\Project\project codes\LGM_pipe.pkl','rb'))
pipe_cat = pickle.load(open('D:\Predictive analysis\Project\project codes\CAT_pipe.pkl','rb'))
pipe_rf = pickle.load(open('D:\Predictive analysis\Project\project codes\RF_pipe.pkl','rb'))
pipe_vot = pickle.load(open('D:\Predictive analysis\Project\project codes\Vote_pipe.pkl','rb'))

teams = ['Chennai Super Kings', 'Royal Challengers Bangalore',
       'Deccan Chargers', 'Kolkata Knight Riders', 'Sunrisers Hyderabad',
       'Kings XI Punjab', 'Delhi Daredevils', 'Mumbai Indians',
       'Pune Warriors', 'Rising Pune Supergiants', 'Rajasthan Royals',
       'Gujarat Lions', 'Kochi Tuskers Kerala']

cities = ['Visakhapatnam', 'Bangalore', 'Cuttack', 'Johannesburg', 'Kolkata',
       'Ahmedabad', 'Jaipur', 'Chennai', 'Chandigarh', 'Hyderabad',
       'Mumbai', 'Centurion', 'Abu Dhabi', 'Pune', 'Delhi', 'Rajkot',
       'Dharamsala', 'Port Elizabeth', 'Ranchi', 'Kimberley', 'Indore',
       'Durban', 'Bloemfontein', 'Raipur', 'East London', 'Cape Town',
       'Kochi', 'Mohali', 'Nagpur', 'Kanpur']

venue= ['Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
       'M Chinnaswamy Stadium', 'Barabati Stadium',
       'New Wanderers Stadium', 'Eden Gardens',
       'Sardar Patel Stadium, Motera', 'Sawai Mansingh Stadium',
       'MA Chidambaram Stadium, Chepauk',
       'Punjab Cricket Association IS Bindra Stadium, Mohali',
       'Rajiv Gandhi International Stadium, Uppal', 'Wankhede Stadium',
       'Brabourne Stadium', 'SuperSport Park', 'Sheikh Zayed Stadium',
       'Maharashtra Cricket Association Stadium',
       'Dubai International Cricket Stadium', 'Feroz Shah Kotla',
       'Saurashtra Cricket Association Stadium',
       'Subrata Roy Sahara Stadium',
       'Himachal Pradesh Cricket Association Stadium', "St George's Park",
       'Dr DY Patil Sports Academy', 'JSCA International Stadium Complex',
       'De Beers Diamond Oval', 'Holkar Cricket Stadium', 'Kingsmead',
       'OUTsurance Oval',
       'Shaheed Veer Narayan Singh International Stadium', 'Buffalo Park',
       'Sharjah Cricket Stadium', 'Newlands', 'Nehru Stadium',
       'Vidarbha Cricket Association Stadium, Jamtha', 'Green Park']

model_dict = {
    'XGBoost Regressor': pipe_xgb,
    'CatBoost Regressor': pipe_cat,
    'Light Gradient Boosting Regressor': pipe_lgbm,    
    'Random Forest': pipe_rf,
    'Voting Regressor': pipe_vot
}

# Define the list of algorithm names for the dropdown
algorithm_names = list(model_dict.keys())
# Algorithms = ['XGBoost','CatBoost','Light Gradient Boost', 'Random Forest', 'Voting Regressor']

st.title('IPL Score Predictor ')

col1, col2 = st.columns(2)

with col1:
    Team_Batting = st.selectbox('Select batting team',sorted(teams))
with col2:
    Team_Bowling = st.selectbox('Select bowling team', sorted(teams))

City_Name = st.selectbox('Select city',sorted(cities))
Venue_Name = st.selectbox('Select venue',sorted(venue))


col3,col4,col5 = st.columns(3)

with col3:
    Current_score = st.number_input('Current Score', min_value=0, step=1)
with col4:
    overs = st.number_input('Overs done(works for over>5)', min_value=0.0, max_value=20.0, step=0.1, format="%.1f")
with col5:
    wickets = st.number_input('Wickets out', min_value=0, max_value=10, step=1)




last_five = st.number_input('Runs scored in last 5 overs',min_value=0, step=1)
Algorithm = st.selectbox('Select Algorithm',algorithm_names)

if st.button('Predict Score'):
    balls_left = 120 - (overs*6)
    wickets_left = 10 -wickets
    crr = Current_score/overs

    input_df = pd.DataFrame(
     {'Team_Batting': [Team_Batting], 'Team_Bowling': [Team_Bowling],'Current_score': [Current_score],'wickets_left': [wickets_left],'crr': [crr],'City_Name':City_Name,'Venue_Name':Venue_Name ,'balls_left': [balls_left],'last_five': [last_five]})
    # result = pipe.predict(input_df)
    # 1. Retrieve the correct model from the dictionary
    current_pipe = model_dict[Algorithm]
    
    # 2. Predict using that specific model
    result = current_pipe.predict(input_df)
    st.header("Predicted Score - " + str(int(result[0])))


