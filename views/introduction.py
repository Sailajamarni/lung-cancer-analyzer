import streamlit as st

def load_view():
    from PIL import Image

    gg = Image.open("images/lung-cancer.jpg")
    st.header('Introduction Of Lung Cancer')

    st.image(gg, caption='Introduction to Lung Cancer',width=600)
    #page title
    st.title('How common is lung cancer?')


    st.write("Lung cancer (both small cell and non-small cell) is the second most common cancer in both men and women in the United States (not counting skin cancer). In men, prostate cancer is more common, while in women breast cancer is more common.")
    # st.markdown(
    # """
    # The American Cancer Society’s estimates for lung cancer in the US for 2023 are:
    # - About 238,340 new cases of lung cancer (117,550 in men and 120,790 in women)
    # - About 127,070 deaths from lung cancer (67,160 in men and 59,910 in women)
    # """
    # )
    new_title = '<p style="font-family:sans-serif; color:cyan; font-size: 20px;">The American Cancer Society’s estimates for lung cancer in the US for 2023 are:- About 238,340 new cases of lung cancer (117,550 in men and 120,790 in women)- About 127,070 deaths from lung cancer (67,160 in men and 59,910 in women)</p>'
    
    st.markdown(new_title, unsafe_allow_html=True)
    st.write("")
    st.title("Is Smoking the only cause ?")
    mawen = Image.open("images/menwa.png")

    st.image(mawen, caption='Smoking is not the major cause',width=650)
    #page title
    
    st.write("The association between air pollution and lung cancer has been well established for decades. The International Agency for Research on Cancer (IARC), the specialised cancer agency of the World Health Organization, classified outdoor air pollution as carcinogenic to humans in 2013, citing an increased risk of lung cancer from greater exposure to particulate matter and air pollution.")




    st.markdown(
    """
    The following list won't indent no matter what I try:
    - A 2012 study by Mumbai’s Tata Memorial Hospital found that 52.1 per cent of lung cancer patients had no history of smoking. 
    - The study contrasted this with a Singapore study that put the number of non-smoking lung cancer patients at 32.5 per cent, and another in the US that found the number to be about 10 per cent.
    - The Tata Memorial study found that 88 per cent of female lung cancer patients were non-smokers, compared with 41.8 per cent of males. It concluded that in the case of non-smokers, environmental and genetic factors were implicated.
    """
    )

    st.title("Not just a Delhi phenomenon ")
    stove = Image.open("images/stove.png")

    st.image(stove, caption='Smoking is not the major cause',width=650)
    #page title
    st.markdown(
    """
    The following list won't indent no matter what I try:
    - In January 2017, researchers at AIIMS, Bhubaneswar, published a demographic profile of lung cancer in eastern India, which found that 48 per cent of patients had not been exposed to active or passive smoking
    - 89 per cent of women patients had never smoked, while the figure for men was 28 per cent.
    - From available research, very little is understood about lung cancer among non-smokers in India. “We need more robust data to identify how strong is the risk and link,” Guleria of AIIMS says.
    """
    )