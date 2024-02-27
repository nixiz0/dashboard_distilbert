import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from create_wordcloud import WordCloud


data = pd.read_csv('data_explo/text_clean.csv')

# WordCloud
st.markdown(f"<h3>WordCloud</h3>", unsafe_allow_html=True)
wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(' '.join(data['text']))
fig, ax = plt.subplots()
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis('off')
st.pyplot(fig)