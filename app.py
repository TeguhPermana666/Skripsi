import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tempfile
from sklearn.neighbors import NearestNeighbors
from vgg16 import VGG_16
from preprocessing import load_and_process_rgb_image
from load_dataset import load_dataset_with_label
from EF_GoggLeNet import GoogLeNet

# Load data
upload= st.file_uploader('Insert image for Recommendation', type=['jpg',"jpeg"])
if upload is not None:
    c1, c2= st.columns(2)
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(upload.read())
    temp_file.close()
    # Get the path for the saved image
    image_path = temp_file.name
    processed_rgb_image = load_and_process_rgb_image(image_path)
    print("Processed RGB image shape:", processed_rgb_image.shape)

    # Classification Step
    load_VGG16_from_weight=VGG_16()
    load_VGG16_from_weight.load_weights(r'Model\Weight-model_VGG16-01.h5')
    load_VGG16_from_weight.compile(loss=keras.losses.categorical_crossentropy,
                optimizer='adam',
                metrics=['accuracy'])
    y_predict=np.argmax(load_VGG16_from_weight.predict(processed_rgb_image))

    # Load the fashion dataset from y_predict result of classification 
    result = []
    load_dataset_with_label(r'Dataset\Dataset_Fashion(11-Label).csv', result, target_label=y_predict)
    labels_fashion, pixels_fashion = result[0]
    print("Prediksi:",y_predict)
    pixels_fashion = pixels_fashion / 255.0

    # Assuming pixels_fashion has shape (num_samples, num_pixels)
    # Determine the size of the images
    image_size = int(np.sqrt(pixels_fashion.shape[1]))
    pixels_fashion = pixels_fashion.reshape(-1, image_size, image_size, 1)
    # Load the GoogLeNet model and modify it for feature extraction
    model_for_feature_extraction = GoogLeNet()

    # Extract features from the Dataset 
    features_fashion = model_for_feature_extraction.predict(pixels_fashion)

    # Ensure the features from the fashion dataset match the shape of the RGB image features
    features_fashion_flat = features_fashion.reshape(features_fashion.shape[0], -1)

    # Create a Nearest Neighbors model
    nn_model = NearestNeighbors(n_neighbors=90, metric='euclidean')
    nn_model.fit(features_fashion_flat)

    # Extract features from the RGB image
    features_rgb_image = model_for_feature_extraction.predict(processed_rgb_image)

    # Flatten the features arrays
    features_rgb_image_flat = features_rgb_image.reshape(1, -1)

    # Reshape input features to match the shape of the training features
    input_features = features_rgb_image_flat

    # Find the k nearest neighbors
    distances, indices = nn_model.kneighbors(input_features)
    indices = indices[0][0:6]
    print("Indices:",indices)
    st.write("Upload Image")
    upload_image = Image.open(upload)
    st.image(upload_image.resize((200, 200)))
    st.write("Recommendation Fashion:")
    col1,col2,col3,col4,col5 = st.columns(5)
    if y_predict == 0:
            with col1:
                st.image(r"Dataset/Ankle Boot/{}.jpg".format(indices[1]))
            with col2:
                st.image(r"Dataset/Ankle Boot/{}.jpg".format(indices[2]))
            with col3:
                st.image(r"Dataset/Ankle Boot/{}.jpg".format(indices[3]))
            with col4:
                st.image(r"Dataset/Ankle Boot/{}.jpg".format(indices[4]))
            with col5:
                st.image(r"Dataset/Ankle Boot/{}.jpg".format(indices[5]))
        
    if y_predict == 1:
            with col1:
                st.image(r"Dataset/Bag/{}.jpg".format(indices[1]))
            with col2:
                st.image(r"Dataset/Bag/{}.jpg".format(indices[2]))
            with col3:
                st.image(r"Dataset/Bag/{}.jpg".format(indices[3]))
            with col4:
                st.image(r"Dataset/Bag/{}.jpg".format(indices[4]))
            with col5:
                st.image(r"Dataset/Bag/{}.jpg".format(indices[5]))
    
    if y_predict == 2:
            with col1:
                st.image(r"Dataset/Coat/{}.jpg".format(indices[1]))
            with col2:
                st.image(r"Dataset/Coat/{}.jpg".format(indices[2]))
            with col3:
                st.image(r"Dataset/Coat/{}.jpg".format(indices[3]))
            with col4:
                st.image(r"Dataset/Coat/{}.jpg".format(indices[4]))
            with col5:
                st.image(r"Dataset/Coat/{}.jpg".format(indices[5]))
                
    if y_predict == 3:
            with col1:
                st.image(r"Dataset/Dress/{}.jpg".format(indices[1]))
            with col2:
                st.image(r"Dataset/Dress/{}.jpg".format(indices[2]))
            with col3:
                st.image(r"Dataset/Dress/{}.jpg".format(indices[3]))
            with col4:
                st.image(r"Dataset/Dress/{}.jpg".format(indices[4]))
            with col5:
                st.image(r"Dataset/Dress/{}.jpg".format(indices[5]))
                
    if y_predict == 4:
            with col1:
                st.image(r"Dataset/Hat/{}.jpg".format(indices[1]))
            with col2:
                st.image(r"Dataset/Hat/{}.jpg".format(indices[2]))
            with col3:
                st.image(r"Dataset/Hat/{}.jpg".format(indices[3]))
            with col4:
                st.image(r"Dataset/Hat/{}.jpg".format(indices[4]))
            with col5:
                st.image(r"Dataset/Hat/{}.jpg".format(indices[5]))
    
    if y_predict == 5:
            with col1:
                st.image(r"Dataset/Sandal/{}.jpg".format(indices[1]),width=120)
            with col2:
                st.image(r"Dataset/Sandal/{}.jpg".format(indices[2]),width=120)
            with col3:
                st.image(r"Dataset/Sandal/{}.jpg".format(indices[3]),width=120)
            with col4:
                st.image(r"Dataset/Sandal/{}.jpg".format(indices[4]),width=120)
            with col5:
                st.image(r"Dataset/Sandal/{}.jpg".format(indices[5]),width=120)
    
    if y_predict == 6:
            with col1:
                st.image(r"Dataset/Shirt/{}.jpg".format(indices[1]))
            with col2:
                st.image(r"Dataset/Shirt/{}.jpg".format(indices[2]))
            with col3:
                st.image(r"Dataset/Shirt/{}.jpg".format(indices[3]))
            with col4:
                st.image(r"Dataset/Shirt/{}.jpg".format(indices[4]))
            with col5:
                st.image(r"Dataset/Shirt/{}.jpg".format(indices[5]))
    
    if y_predict == 7:
            with col1:
                st.image(r"Dataset/Sneaker/{}.jpg".format(indices[1]))
            with col2:
                st.image(r"Dataset/Sneaker/{}.jpg".format(indices[2]))
            with col3:
                st.image(r"Dataset/Sneaker/{}.jpg".format(indices[3]))
            with col4:
                st.image(r"Dataset/Sneaker/{}.jpg".format(indices[4]))
            with col5:
                st.image(r"Dataset/Sneaker/{}.jpg".format(indices[5]))
    
    if y_predict == 8:
            with col1:
                st.image(r"Dataset/Trousers/{}.jpg".format(indices[1]))
            with col2:
                st.image(r"Dataset/Trousers/{}.jpg".format(indices[2]))
            with col3:
                st.image(r"Dataset/Trousers/{}.jpg".format(indices[3]))
            with col4:
                st.image(r"Dataset/Trousers/{}.jpg".format(indices[4]))
            with col5:
                st.image(r"Dataset/Trousers/{}.jpg".format(indices[5]))   
    
    if y_predict == 9:
            with col1:
                st.image(r"Dataset/Tshirt_Top/{}.jpg".format(indices[1]))
            with col2:
                st.image(r"Dataset/Tshirt_Top/{}.jpg".format(indices[2]))
            with col3:
                st.image(r"Dataset/Tshirt_Top/{}.jpg".format(indices[3]))
            with col4:
                st.image(r"Dataset/Tshirt_Top/{}.jpg".format(indices[4]))
            with col5:
                st.image(r"Dataset/Tshirt_Top/{}.jpg".format(indices[5]))
    
    if y_predict == 10:
            with col1:
                st.image(r"Dataset/Pullover/{}.jpeg".format(indices[1]))
            with col2:
                st.image(r"Dataset/Pullover/{}.jpeg".format(indices[2]))
            with col3:
                st.image(r"Dataset/Pullover/{}.jpeg".format(indices[3]))
            with col4:
                st.image(r"Dataset/Pullover/{}.jpeg".format(indices[4]))
            with col5:
                st.image(r"Dataset/Pullover/{}.jpeg".format(indices[5]))