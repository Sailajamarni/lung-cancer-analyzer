import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tempfile import NamedTemporaryFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import img_to_array

def load_view():
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)  

    st.set_option('deprecation.showfileUploaderEncoding', False)
    
    @st.cache(allow_output_mutation=True)
    def loading_model():
        fp = "models/keras_model.h5"
        model_loader = load_model(fp)
        return model_loader

    cnn = loading_model()
    st.write("""# Lung Cancer Detection using CNN and CT-Scan Images""")

    temp = st.file_uploader("Upload CT-Scan Image", type=['png', 'jpeg', 'jpg'])
    if temp is not None:
        file_details = {"FileName": temp.name, "FileType": temp.type, "FileSize": temp.size}
        st.write(file_details)

    buffer = temp
    temp_file = NamedTemporaryFile(delete=False)
    if buffer:
        temp_file.write(buffer.getvalue())
        st.write(image.load_img(temp_file.name))

    if buffer is None:
        st.text("Please upload an image file")
    else:
        ved_img = image.load_img(temp_file.name, target_size=(224, 224))
        pp_ved_img = img_to_array(ved_img)
        pp_ved_img = pp_ved_img / 255
        pp_ved_img = np.expand_dims(pp_ved_img, axis=0)

        # predict
        hardik_preds = cnn.predict(pp_ved_img)
        print(hardik_preds[0])
        if hardik_preds[0][0] >= 0.5:
            out = ('I am {:.2%} percent confirmed that this is a Normal Case'.format(hardik_preds[0][0]))
            st.balloons()
            st.success(out)
        else: 
            out = ('I am {:.2%} percent confirmed that this is a Lung Cancer Case'.format(1 - hardik_preds[0][0]))
            st.error(out)

        img_display = Image.open(temp_file.name)
        st.image(img_display, use_column_width=True)

# Call the load_view function
