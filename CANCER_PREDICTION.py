import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog

# Load the pre-trained model
model = tf.keras.models.load_model ( 'cancer2.keras' )


# Function to test a single image
def test_single_image ( image_path ) :
    img_size = 150
    # Read and preprocess the image
    img = cv2.imread ( image_path )
    img = cv2.cvtColor ( img , cv2.COLOR_BGR2RGB )
    img = cv2.resize ( img , (img_size , img_size) )
    x = np.array ( img )
    x = x / 255
    x = np.expand_dims ( x , axis = 0 )
    # Make prediction
    pred = model.predict ( x )
    predicted_class = np.argmax ( pred , axis = 1 ) [ 0 ]
    # Get the category using inverse transform
    category = [ "benign" , "malignant" ]
    predicted_category = category [ predicted_class ]
    return img , predicted_category


root = tk.Tk ( )
root.withdraw ( )
filepaths = filedialog.askopenfilenames ( )  # Allow selecting multiple files

for filepath in filepaths :
    img , predicted_category = test_single_image ( filepath )

    # Display the result
    plt.imshow ( img )
    plt.title ( f"Predicted category: {predicted_category}" )
    plt.show ( )
