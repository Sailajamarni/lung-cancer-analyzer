# import base64
import pickle
import streamlit as st

import utils as utl
from views import introduction,about,prediction,cnn
 

st.set_page_config(page_title='Lung Cancer Detection')
#Loading models
cancer_model = pickle.load(open('models/final_model.sav', 'rb'))


st.set_option('deprecation.showPyplotGlobalUse', False)
utl.inject_custom_css()
utl.navbar_component()


def navigation():
    route = utl.get_current_route()
    if route == "introduction":
        introduction.load_view()
    elif route == "about_dataset":
        about.load_view()
    elif route == "prediction":
        prediction.load_view()
    elif route == "cnn_based":
        cnn.load_view()
    elif route == None:
        introduction.load_view()
        
navigation()