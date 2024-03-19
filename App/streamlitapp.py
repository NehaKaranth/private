import streamlit as st
import os 
import imageio 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf 
from utils import load_data, convert_to_mp4, num_to_char, char_to_num
from modelutil import load_model
from sklearn.metrics import classification_report, confusion_matrix

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('LipNet App') 
# Generating a list of options or videos 
options = ('../data/s1/')

options=os.listdir(options)
selected_video = st.selectbox('Choose video', options)
print('selected video',selected_video)

file_path = os.path.join('../data/s1/', selected_video)
video, annotations = load_data(tf.convert_to_tensor(file_path))
# Generate two columns
col1, col2 = st.columns(2)

if options:

    # Rendering the video
    with col1:
        st.info('The video below displays the converted video in mp4 format')
        mp4_path = 'test_video.mp4'
        # Convert the video
        convert_to_mp4(file_path, mp4_path)

        # Rendering inside of the app
        if os.path.exists(mp4_path):
            with open(mp4_path, 'rb') as video_file:
                video_bytes = video_file.read()
                st.video(video_bytes)
            st.info('Original tokens/words:')
            original_token = tf.strings.reduce_join(num_to_char(annotations)).numpy().decode('utf-8')
            st.text(original_token)

        else:
            st.error("Failed to convert the video. Please check the input file path.")

    with col2:
        st.info('This is all the machine learning model sees when making a prediction')
        frame = np.array(video)
        frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame)) * 255
        frame = frame.astype(np.uint8)
        frame = np.squeeze(frame, axis=-1)
        imageio.mimsave('animation.gif', frame, duration=100)
        st.image('animation.gif', width=400)

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        predicted_token = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(predicted_token)

st.subheader('Evaluation Metrics')
true_labels = list(original_token.split(" "))
predicted_labels = list(predicted_token.split(" "))

cls_report = classification_report(true_labels, predicted_labels, target_names=true_labels)

# Compute confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Display classification report
st.info('Classification Report:')
st.text(cls_report)

# Display confusion matrix
st.info('Confusion Matrix:')
fig, ax = plt.subplots(figsize=(8, 6))  
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', ax=ax) 
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
st.pyplot(fig) 