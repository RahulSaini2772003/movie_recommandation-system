import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64


df = pd.read_csv('imdb_top_1000.csv')

df['Certificate'] = df['Certificate'].fillna(df['Certificate'].mode()[0])
df['Meta_score'] = df['Meta_score'].fillna(df['Meta_score'].mean())
df['Gross'] = pd.to_numeric(df['Gross'].str.replace(",", ""))
df['Gross'] = df['Gross'].fillna(df['Gross'].mean())

def clean_duration(duration): 
    if isinstance(duration, str):
        if 'min' in duration:
            try:
                return float(duration.replace('min', '').strip())
            except ValueError:
                return np.nan
        else:
            return np.nan
    return duration 

df['Runtime'] = df['Runtime'].apply(clean_duration)
df['Runtime'] = df['Runtime'].fillna(df['Runtime'].mean())
df['Genre'] = df['Genre'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])
df['ReleaseYear'] = df['Released_Year'].astype(str)

df['combined_features'] = df['Series_Title'] + ' ' + df['Certificate'] + ' ' + df['IMDB_Rating'].astype(str) + ' ' + \
                            df['Genre'].apply(lambda x: ' '.join(x)) + ' ' + df['ReleaseYear'] + ' ' + df['Runtime'].astype(str) + \
                            ' ' + df['Overview'].astype(str) + ' ' + df['Meta_score'].astype(str) + \
                            ' ' + df['Director'].astype(str) + ' ' + df['Star1'].astype(str) + ' ' + df['Star2'].astype(str) + \
                            ' ' + df['Star3'].astype(str) + ' ' + df['Star4'].astype(str) + ' ' + df['No_of_Votes'].astype(str) + ' ' + \
                            df['Gross'].astype(str)

simple_df = df[['Poster_Link','Series_Title', 'combined_features','IMDB_Rating','Runtime','Genre','Overview']]
cv = CountVectorizer()
feature_matrix = cv.fit_transform(simple_df['combined_features'])
cosine_sim = cosine_similarity(feature_matrix, feature_matrix)
indices = pd.Series(simple_df.index, index=simple_df['Series_Title']).to_dict()


def recommend_movies(title, cosine_sim, df, indices):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:7] 
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices][['Series_Title', 'Poster_Link','combined_features','IMDB_Rating','Runtime','Genre','Overview']]


def main():
    st.set_page_config(layout="wide")
    @st.cache_data()
    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

    png_file = '5.jpg'
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: linear-gradient(to bottom right, rgba(0,0,0,0.88), rgba(0,0,0,0.88)), url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        height: 100vh;
        width: 100vw; 
        position: fixed;
        top: 0;
        left: 0;
    }}
    .centered-title {{
        text-align: center;
        margin: 0 auto;
        color: white;
    }}
    .stSelectbox{{
        margin: 0 auto !important;
        width: 60vw !important;
    }}
    .stButton button {{
        margin: 0 15vw !important;
        display: block;
        }}
    .stTitle title{{
        margin: 0 auto !important;
        }}
    </style>'''
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
    st.markdown('<div class="centered-title"><h1>Movie Recommendation System</h1></div>', unsafe_allow_html=True)
    
    # st.title('Movie Recommendation System')
    
    movie_title = st.selectbox('Select a Movie', simple_df['Series_Title'])

    if st.button('Recommend'):
        recommended_movies = recommend_movies(movie_title, cosine_sim, simple_df, indices)
        st.subheader(f'Movies Similar to {movie_title}:')
        st.write("")

        # Create a row of columns for recommended movies
        cols = st.columns([10, 10, 10, 10, 10, 10])
        for idx, (i, row) in enumerate(recommended_movies.iterrows()):
            with cols[idx]:
                st.image(row['Poster_Link'], use_column_width=True)
                st.markdown(f"""
                    <div style="font-size:26px; font-weight:bold;">{row['Series_Title']}</div>
                    <div style="font-size:15px; color:#edbcf9;">IMDB Rating: {row['IMDB_Rating']}</div>
                    <div style="font-size:15px; color:#7fafd6;">Runtime: {row['Runtime']} min</div>
                    <div style="font-size:15px; color:orange;">Genre: {', '.join(row['Genre'])}</div>
                """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()

