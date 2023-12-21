import numpy as np
import cv2
from tensorflow.keras.models import load_model
import streamlit as st
model = load_model("model.h5")

labels = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']

detectedTumor = None

def RunModel(file_path):
    img = cv2.imread(file_path)
    if img is None:
        print("Error: Failed to read the image file.")
        return
    img = cv2.resize(img, (150, 150))
    print("Printing Image:", img)
    img_array = np.array(img)
    print("NumpyArray Image:", img_array)

    img_array = img_array.reshape(1, 150, 150, 3)
    print(img_array.shape)

    print("Before Model -----------------------")
    a = model.predict(img_array)
    print("After Model -----------------------")
    indices = a.argmax()
    print("Index: ", indices)
    if indices == 2:
        st.write("No Cancer Detected...")
    else:
        st.write("Cancer Detected...")
        detectedTumor = labels[indices]
        st.write("Tumor Type: ", labels[indices])
        # return detectedTumor


import os
import tempfile

def main():
    st.title("Brain Cancer Detection and Classification")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Create a temporary directory
        temp_dir = tempfile.TemporaryDirectory()
        temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)

        # Save the uploaded file to the temporary directory
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Print the image path
        # st.write("Image Path:", temp_file_path)

        # detectedTumor_ = None
        # Run Model
        if st.button("Run Model"):
            # Run Model
            RunModel(temp_file_path)

            # if detectedTumor_:
            #     if st.button("Tumor Type"):
            #         st.write("Tumor Type: ", detectedTumor_)
            # st.write("---------------Execution Completed-------------")
        # RunModel(temp_file_path)

        # Cleanup the temporary directory
        temp_dir.cleanup()

        # if st.button("Tumor Type"):
        #     st.write("Tumor Type: ", detectedTumor_)

if __name__ == "__main__":
    main()