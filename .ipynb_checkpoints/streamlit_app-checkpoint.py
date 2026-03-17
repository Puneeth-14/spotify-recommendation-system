import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv(r"spotify_songs.csv")

features = [
'danceability','energy','loudness','speechiness',
'acousticness','instrumentalness','liveness','valence','tempo'
]

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

similarity = cosine_similarity(X_scaled)

def recommend(song_name):

    idx = df[df['track_name'] == song_name].index[0]

    distances = similarity[idx]

    songs = list(enumerate(distances))
    songs = sorted(songs, key=lambda x: x[1], reverse=True)[1:6]

    results = []

    for i in songs:
        results.append({
            "song": df.iloc[i[0]].track_name,
            "artist": df.iloc[i[0]].track_artist,
            "genre": df.iloc[i[0]].playlist_genre
        })

    return results


st.title("🎵 Spotify Recommendation System")

song = st.selectbox("Select a Song", df['track_name'].values)

if st.button("Recommend Songs"):

    recommendations = recommend(song)

    for r in recommendations:

        st.write("🎧 **Song:**", r["song"])
        st.write("👨‍🎤 **Artist:**", r["artist"])
        st.write("🎼 **Genre:**", r["genre"])
        st.write("---") 

st.title("Spotify Recommendation System")