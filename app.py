import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBRegressor

pipe = pickle.load(open('pipe.pkl','rb'))

teams = ['Peshawar Zalmi', 'Quetta Gladiators', 'Karachi Kings',
       'Islamabad United', 'Lahore Qalandars', 'Multan Sultans']

cities = ['Dubai', 'Sharjah', 'Lahore', 'Karachi', 'Abu Dhabi', 'Multan',
       'Rawalpindi']

st.title('PSL Match Score Predictor')


# Create a grid of two columns
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select bowling team', sorted(teams))

# Create a grid of three columns
col3, col4, col5 = st.columns(3)

with col3:
    current_score = st.number_input('Current Score', key='current_score')
with col4:
    overs = st.number_input('Overs done(works for over>5)', key='overs')
with col5:
    wickets = st.number_input('Wickets out', key='wickets')

city = st.selectbox('Select city', sorted(cities), key='city')
last_five = st.number_input('Runs scored in last 5 overs', key='last_five')

#(col1, col2 = st.beta_columns(2)

#with col1:
    #batting_team = st.selectbox('Select batting team',sorted(teams))
#with col2:
    #bowling_team = st.selectbox('Select bowling team', sorted(teams))

#city = st.selectbox('Select city',sorted(cities))

#col3,col4,col5 = st.beta_columns(3)

#with col3:
    #current_score = st.number_input('Current Score')
#with col4:
    #overs = st.number_input('Overs done(works for over>5)')
#with col5:
    #wickets = st.number_input('Wickets out')

if st.button('Predict Score'):
    balls_left = 120 - (overs*6)
    wickets_left = 10 -wickets
    crr = current_score/overs

    input_df = pd.DataFrame(
     {'batting_team': [batting_team], 'bowling_team': [bowling_team],'city':city, 'current_score': [current_score],'balls_left': [balls_left], 'wickets_left': [wickets], 'crr': [crr], 'last_five': [last_five]})
    result = pipe.predict(input_df)
    st.header("Predicted Score - " + str(int(result[0])))



