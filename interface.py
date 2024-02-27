import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tensorflow as tf
from collections import Counter
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from tensorflow.keras.models import load_model
from transformers import DistilBertTokenizer


# To run Locally the streamlit app :
# <= streamlit run interface.py =>

# User can select the text size
size = st.selectbox('Select text size', ['default', 'Large'])

if size == 'default':
    font_size = '23px'
elif size == 'Large':
    font_size = '28px'

data = pd.read_csv('data_explo/text_clean.csv')

st.markdown(f"<h2 style='font-size: {font_size}; color: white;'>Exploratory analysis of text data</h2>", unsafe_allow_html=True)

# Show the first rows of the DataFrame
st.write(data.head())

# Create a dictionary to map numbers to emotion names
emotion_dict = {0: 'Sadness', 1: 'Joy', 2: 'Love', 3: 'Anger', 4: 'Fear', 5: 'Surprise'}

st.markdown(f"<h4 style='font-size: {font_size}; color: white;'>Emotions legend :</h4>", unsafe_allow_html=True)
st.markdown(f"<div style='font-size: {font_size}; color: white;'>- sadness : 0</div>", unsafe_allow_html=True)
st.markdown(f"<div style='font-size: {font_size}; color: white;'>- joy : 1</div>", unsafe_allow_html=True)
st.markdown(f"<div style='font-size: {font_size}; color: white;'>- love : 2</div>", unsafe_allow_html=True)
st.markdown(f"<div style='font-size: {font_size}; color: white;'>- anger : 3</div>", unsafe_allow_html=True)
st.markdown(f"<div style='font-size: {font_size}; color: white;'>- fear : 4</div>", unsafe_allow_html=True)
st.markdown(f"<div style='font-size: {font_size}; color: white;'>- surprise : 5</div>", unsafe_allow_html=True)

@st.cache_data
def preprocess_data(data):
    # Replace numbers with names in the DataFrame
    data['label'] = data['label'].map(emotion_dict)
    # Calculate the number of each label
    label_counts = data['label'].value_counts().sort_values(ascending=False)
    # Sentence size analysis
    data['text_length'] = data['text'].apply(len)
    return data, label_counts

data, label_counts = preprocess_data(data)

# Analysis of the distribution of labels
st.markdown(f"<h3 style='font-size: {font_size}; color: white; text-align: center;'>Analysis of the distribution of labels</h3>", unsafe_allow_html=True)

# Create the chart
fig = px.bar(label_counts, x=label_counts.index, y=label_counts.values, color_discrete_sequence=px.colors.qualitative.Pastel)
st.plotly_chart(fig)

# Sentence size analysis
st.markdown(f"<h3 style='font-size: {font_size}; color: white; text-align: center;'>Sentence size</h3>", unsafe_allow_html=True)
fig = px.histogram(data, x='text_length', color_discrete_sequence=px.colors.qualitative.Pastel)
st.plotly_chart(fig)

# Word frequency analysis
st.markdown(f"<h3 style='font-size: {font_size}; color: white; text-align: center;'>Word frequency</h3>", unsafe_allow_html=True)
words = ' '.join(data['text']).split()
word_freq = Counter(words)
word_freq_df = pd.DataFrame.from_dict(word_freq, orient='index').reset_index()
word_freq_df.columns = ['Word', 'Frequency']
top_words = word_freq_df.sort_values(by='Frequency', ascending=False).head(20)
fig = px.bar(top_words, x='Frequency', y='Word', orientation='h', color_discrete_sequence=px.colors.qualitative.Pastel)
st.plotly_chart(fig)

# WordCloud
st.markdown(f"<h3 style='font-size: {font_size}; color: white; text-align: center;'>WordCloud</h3>", unsafe_allow_html=True)
st.image('wordcloud_img.png')

st.markdown("---")

load_dotenv('.azure_secret')
azure_storage_account_key = os.getenv('AZURE_BLOB_KEY')
azure_storage_account_name = "distilbert"
container_name = "modeldistilbert"

def download_from_azure_storage():
    blob_service_client = BlobServiceClient.from_connection_string(
        f"DefaultEndpointsProtocol=https;AccountName={azure_storage_account_name};AccountKey={azure_storage_account_key}"
    )
    container_client = blob_service_client.get_container_client(container_name)
    
    # Create the folder if it does not already exist
    os.makedirs('distilbert', exist_ok=True)
    os.makedirs('distilbert/variables', exist_ok=True)
    
    # Iterate over each blob in the container and download it
    for blob in container_client.list_blobs():
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob.name)
        # Check if the blob is in the 'distilbert/variables' folder
        if 'distilbert/variables' in blob.name:
            # Get the file name after the 'distilbert/variables/' part
            split_name = blob.name.split('distilbert/variables/')
            if len(split_name) > 1:
                file_name = split_name[1]
                with open(os.path.join('distilbert/variables', file_name), "wb") as download_file:
                    download_file.write(blob_client.download_blob().readall())
        # Check if the blob is in the 'distilbert' folder but not in the 'variables' subfolder
        elif 'distilbert' in blob.name and 'variables' not in blob.name:
            # Get the file name after the 'distilbert/' part
            split_name = blob.name.split('distilbert/')
            if len(split_name) > 1:
                file_name = split_name[1]
                with open(os.path.join('distilbert', file_name), "wb") as download_file:
                    download_file.write(blob_client.download_blob().readall())

download_from_azure_storage()

# Load tokenizer_distilbert and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = load_model("distilbert")

# Emotion prediction
st.markdown(f"<h2>Emotion prediction</h2>", unsafe_allow_html=True)
user_input = st.text_input(f"Enter your text here")
if user_input:
    # Encode the user input and get both input_ids and attention_mask
    user_input_vectorized = tokenizer(user_input, return_tensors='tf', padding=True, truncation=True, max_length=512)
    
    # Predict emotion & Get the predicted class
    prediction = model(user_input_vectorized)
    predicted_class_indices = np.argmax(prediction['logits'], axis=1)

    # Transform class indices to class names
    predicted_class_names = [emotion_dict[i] for i in predicted_class_indices]
    
    # Show predicted emotion
    st.write(f"The predicted emotion is : {predicted_class_names}")