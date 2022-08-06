#!/usr/bin/env python
# coding: utf-8

# In[5]:

import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
from tensorflow.keras.preprocessing import image as IMG


st.image("https://www.movassat.de/faznet-aus-deutschland-abgeschobener-afghane-toetet-sich-selbst", width=200)
st.image("https://www.designtagebuch.de/rebrush-bei-spiegel-online/spiegel-online-logo/", width=200)
st.header("News Image Classifier - F.A.Z. vs. Spiegel")


file_uploaded = st.file_uploader("Choose the file", type = ['jpg', 'png', 'jpeg'])
if file_uploaded is not None:
	image = Image.open(file_uploaded)
	figure = plt.figure()
	plt.imshow(image)
	plt.axis('off')
	st.pyplot(figure)
	img = IMG.load_img(file_uploaded,target_size=[150,150])


	classifier = tf.keras.models.load_model(r'vgg16_model_array_fazsp.h5')
	x = IMG.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	class_names = ['FAZ', 'Spiegel']
	pred = classifier.predict(x)
	scores = tf.nn.softmax(pred[0])
	scores = scores.numpy()
	image_class = class_names[np.argmax(scores)]
	st.write("The image uploaded is: {}".format(image_class))
