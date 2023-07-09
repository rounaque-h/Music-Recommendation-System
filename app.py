import pickle

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tqdm import tqdm

df = pd.read_csv("data.csv")

df["artists"]=df["artists"].str.replace("[", "")
df["artists"]=df["artists"].str.replace("]", "")
df["artists"]=df["artists"].str.replace("'", "")

temp=df['id']
df["song"]='https://open.spotify.com/track/' + temp.astype(str)

def convert(row):
    return '<a href="{}" target="_blank">{}</a>'.format(row['song'],  row[12])
df['song'] = df.apply(convert, axis=1)

class SpotifyRecommender():
    def __init__(self, rec_data):
        #our class should understand which data to work with
        self.rec_data_ = rec_data

    #if we need to change data
    def change_data(self, rec_data):
        self.rec_data_ = rec_data

    #function which returns recommendations, we can also choose the amount of songs to be recommended
    def get_recommendations(self, song_name, amount=1):
        distances = []
        #choosing the data for our song
        song = self.rec_data_[(self.rec_data_.name.str.lower() == song_name.lower())].head(1).values[0]
        #dropping the data with our song
        res_data = self.rec_data_[self.rec_data_.name.str.lower() != song_name.lower()]
        for r_song in tqdm(res_data.values):
            dist = 0
            for col in np.arange(len(res_data.columns)):
                #indeces of non-numerical columns
                if not col in [1, 6, 12, 14, 18, 19]:
                    #calculating the manhettan distances for each numerical feature
                    dist = dist + np.absolute(float(song[col]) - float(r_song[col]))
            distances.append(dist)
        res_data['distance'] = distances
        #sorting our data to be ascending by 'distance' feature
        res_data = res_data.sort_values('distance')
        columns = ['artists', 'song']
        return res_data[columns][:amount]

recommender = SpotifyRecommender(df)

def predict_mrs(value,no_of_r):
  st.write(recommender.get_recommendations(value, int(no_of_r)).to_html(escape=False, index=False), unsafe_allow_html=True)

pickle_out = open("predict_mrs.pkl", "wb")
pickle.dump(predict_mrs, pickle_out)
pickle_out.close()

pickle_in = open('predict_mrs.pkl', 'rb')
classifier = pickle.load(pickle_in)


st.title('Music Recommendation System')
st.subheader('Song Name:')
song_name = st.text_input('')
submit = st.button('Predict')


# Slider
st.subheader("No of Recommendations:")
st.markdown(
    f'''
        <style>
            .sidebar .sidebar-content {{
                width: 700px;
            }}
        </style>
    ''',
    unsafe_allow_html=True
)

no_of_r = st.slider("", 1, 10)
st.subheader('Selected: {}'.format(no_of_r))

if submit:
    predict_mrs(song_name,no_of_r)

# SIDEBAR
image1 = Image.open('images/logo.png')
st.sidebar.image(image1)

st.sidebar.title('ABOUT')
st.sidebar.write('* Music Recommendation system that recommends similar songs as searched by user. Developed by Rohit Kumar Bid, Md Rounaque Afroz Haider, Nishi Kumari, Sachin Kumar, Prakash Soren')
st.sidebar.write('* Recommendation engine implementing K-means clusterinig using Manhattan distance that uses charecteristics such as artist, release date, acousticness, tempo, etc.')