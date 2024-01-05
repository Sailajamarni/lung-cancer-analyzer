import streamlit as st
import pickle
import numpy as np
import pandas as pd

cancer_model = pickle.load(open('models/final_model.sav', 'rb'))
    
def load_view():
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)  
    
    #page title
    
    ##this code is reduce effort of manuallu entering values
    testx=pd.read_csv("datasets/testx.csv",index_col=0)
    testy=pd.read_csv("datasets/testy.csv",index_col=0)
    testx.reset_index(drop=True, inplace=True)
    testy.reset_index(drop=True, inplace=True)
    
    concate_data = pd.concat([testx,testy],axis=1)

    st.title('Lung Cancer Prediction using ML')

    idn = st.slider('Select any index from Testing Data', 0, 200, 25)
    a=concate_data.iloc[idn]
    st.write("Displaying vales of index ", idn)
    aa=list(concate_data.iloc[idn])
    if st.button('Show me this value'):
        st.write(aa)

    ##values will come directly from here no manual needed
    a=concate_data.iloc[idn][0]
    b=concate_data.iloc[idn][1]
    c=concate_data.iloc[idn][2]
    d=concate_data.iloc[idn][3]
    e=concate_data.iloc[idn][4]
    f=concate_data.iloc[idn][5]
    g=concate_data.iloc[idn][6]
    h=concate_data.iloc[idn][7]
    i=concate_data.iloc[idn][8]
    j=concate_data.iloc[idn][9]
    k=concate_data.iloc[idn][10]
    l=concate_data.iloc[idn][11]
    m=concate_data.iloc[idn][12]
    n=concate_data.iloc[idn][13]
    o=concate_data.iloc[idn][14]
    p=concate_data.iloc[idn][15]
    q=concate_data.iloc[idn][16]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        Age = st.text_input('Age', key="1",value=a)
        if len(Age)==0:
            Age=0
    with col2:
        Gender = st.text_input('Gender', key="2",value=b)
        if len(Gender)==0:
            Gender=0
    with col3:
        AirPollution = st.text_input('Air Pollution', key="3",value=c)
        if len(AirPollution)==0:
            AirPollution=0
    with col1:
        Alcoholuse = st.text_input('Alcohol Use', key="4",value=d)  
        if len(Alcoholuse)==0:
            Alcoholuse=0
    with col2:
        BalancedDiet = st.text_input('Balanced Diet', key="5",value=e)
        if len(BalancedDiet)==0:
            BalancedDiet=0
    with col3:
        Obesity = st.text_input('Obesity', key="6",value=f)
        if len(Obesity)==0:
            Obesity=0
    with col1:
        Smoking = st.text_input('Smoking', key="7",value=g)
        if len(Smoking)==0:
            Smoking=0
    with col2:
        PassiveSmoker = st.text_input('Passive Smoker', key="8",value=h)
        if len(PassiveSmoker)==0:
            PassiveSmoker=0
    with col3:
        Fatigue = st.text_input('Fatigue', key="9",value=i)
        if len(Fatigue)==0:
            Fatigue=0
    with col1:
        WeightLoss = st.text_input('Weight Loss', key="10",value=j)
        if len(WeightLoss)==0:
            WeightLoss=0
    with col2:
        ShortnessofBreath = st.text_input('Shortness of Breath', key="11",value=k)
        if len(ShortnessofBreath)==0:
            ShortnessofBreath=0
    with col3:
        Wheezing = st.text_input('Wheezing', key="12",value=l)
        if len(Wheezing)==0:
            Wheezing=0
    with col1:
        SwallowingDifficulty = st.text_input('Swallowing Difficulty', key="13",value=m)
        if len(SwallowingDifficulty)==0:
            SwallowingDifficulty=0
    with col2:
        ClubbingofFingerNails = st.text_input('Clubbing of Finger Nails', key="14",value=n)
        if len(ClubbingofFingerNails)==0:
            ClubbingofFingerNails=0
    with col3:
        FrequentCold = st.text_input('Frequent Cold', key="15",value=o)
        if len(FrequentCold)==0:
            FrequentCold=0
    with col1:
        DryCough = st.text_input('Dry Cough', key="16",value=p)    
        if len(DryCough)==0:
            DryCough=0
    with col2:
        Snoring = st.text_input('Snoring  ', key="17",value=q)
        if len(Snoring)==0:
            Snoring=0
#     if(len(Age)==None):
#             st.warning('enter age', icon="⚠")
#             # Validate and convert input values to floats
#     try:
#         Age = float(Age) if Age and Age.strip() else None
#         Gender = float(Gender) if Gender and Gender.strip() else None
#         AirPollution = float(AirPollution) if AirPollution and AirPollution.strip() else None
#         Alcoholuse = float(Alcoholuse) if Alcoholuse and Alcoholuse.strip() else None
#         BalancedDiet = float(BalancedDiet) if BalancedDiet and BalancedDiet.strip() else None
#         Obesity = float(Obesity) if Obesity and Obesity.strip() else None
#         Smoking = float(Smoking) if Smoking and Smoking.strip() else None
#         PassiveSmoker = float(PassiveSmoker) if PassiveSmoker and PassiveSmoker.strip() else None
#         Fatigue = float(Fatigue) if Fatigue and Fatigue.strip() else None
#         WeightLoss = float(WeightLoss) if WeightLoss and WeightLoss.strip() else None
#         ShortnessofBreath = float(ShortnessofBreath) if ShortnessofBreath and ShortnessofBreath.strip() else None
#         Wheezing = float(Wheezing) if Wheezing and Wheezing.strip() else None
#         SwallowingDifficulty = float(SwallowingDifficulty) if SwallowingDifficulty and SwallowingDifficulty.strip() else None
#         ClubbingofFingerNails = float(ClubbingofFingerNails) if ClubbingofFingerNails and ClubbingofFingerNails.strip() else None
#         FrequentCold = float(FrequentCold) if FrequentCold and FrequentCold.strip() else None
#         DryCough = float(DryCough) if DryCough and DryCough.strip() else None
#         Snoring = float(Snoring) if Snoring and Snoring.strip() else None
#     except ValueError:
#         st.warning("Invalid input. Please enter valid numerical values for all columns.")
#     # Stop further processing if there's an invalid input
#         st.stop()

# # Check if any value is not entered in columns
#     if any(value is None for value in [Age, Gender, AirPollution, Alcoholuse, BalancedDiet, Obesity, Smoking, PassiveSmoker, Fatigue, WeightLoss, ShortnessofBreath, Wheezing, SwallowingDifficulty,  ClubbingofFingerNails, FrequentCold, DryCough, Snoring]):
#         st.warning("Please enter values for all columns.")
#     else:
#     # Make prediction only if all values are valid
#         heart_prediction = cancer_model.predict([[Age, Gender, AirPollution, Alcoholuse, BalancedDiet, Obesity, Smoking, PassiveSmoker, Fatigue, WeightLoss, ShortnessofBreath, Wheezing, SwallowingDifficulty, ClubbingofFingerNails, FrequentCold, DryCough, Snoring]])
#     # Rest of your code for handling the prediction result

 
    # code for Prediction
    heart_diagnosis = ''
   # for column_name in heart_prediction:
      #      if len(heart_prediction[column_name]) == 0:
     #            print(f"Enter data for column: {column_name}")
    # creating a button for Prediction
    
    if st.button('Lung Disease Test Result'):
        heart_prediction = cancer_model.predict([[Age, Gender, AirPollution, Alcoholuse, BalancedDiet, Obesity, Smoking, PassiveSmoker, Fatigue, WeightLoss,ShortnessofBreath, Wheezing, SwallowingDifficulty,ClubbingofFingerNails, FrequentCold, DryCough, Snoring]])
        if(Age==0 or Gender==0 or AirPollution==0 or Alcoholuse==0 or BalancedDiet==0 or Obesity==0 or Smoking==0 or PassiveSmoker==0 or Fatigue==0 or WeightLoss==0 or ShortnessofBreath==0 or Wheezing==0 or SwallowingDifficulty==0 or ClubbingofFingerNails==0 or FrequentCold==0 or DryCough==0 or Snoring==0):
            fields = ['Age', 'Gender', 'AirPollution', 'Alcoholuse', 'BalancedDiet', 'Obesity', 'Smoking', 'PassiveSmoker', 'Fatigue', 'WeightLoss', 'ShortnessofBreath', 'Wheezing', 'SwallowingDifficulty', 'ClubbingofFingerNails', 'FrequentCold', 'DryCough', 'Snoring']

            missing_fields = []

            for field in fields:
                if eval(field) == 0:
                    missing_fields.append(field)

            if missing_fields:
                error_message = f"Please enter values for the following fields: {', '.join(missing_fields)}"
                st.error(error_message)                    
        elif(heart_prediction[0] == 'High'):
            heart_diagnosis = 'The person is having lung cancer'
            st.error(heart_diagnosis)

        elif(heart_prediction[0] == 'Medium'):
          heart_diagnosis = 'The person is chance of having lung cancer'
          st.warning(heart_diagnosis)
        else:
          heart_diagnosis = 'The person does not have any lung disease'
          st.balloons()
          st.success(heart_diagnosis)
#     if '' in [Age, Gender, AirPollution, Alcoholuse, BalancedDiet, Obesity, Smoking, PassiveSmoker, 
#            Fatigue, WeightLoss, ShortnessofBreath, Wheezing, SwallowingDifficulty, 
#            ClubbingofFingerNails, FrequentCold, DryCough, Snoring]:
#             st.warning("Please enter values for all columns.")
#     else:
#     # Convert valid values to float
#         Age = float(Age) if Age else None
#     # Repeat this for other columns

#     # Make prediction only if all values are valid
#     if all(value is not None for value in [Age, Gender, AirPollution, Alcoholuse, BalancedDiet, Obesity, Smoking, PassiveSmoker, Fatigue, WeightLoss, ShortnessofBreath, Wheezing, SwallowingDifficulty, ClubbingofFingerNails, FrequentCold, DryCough, Snoring]):
#         heart_prediction = cancer_model.predict([[Age, Gender, AirPollution, Alcoholuse, BalancedDiet, Obesity, Smoking, PassiveSmoker, Fatigue, WeightLoss, ShortnessofBreath, Wheezing, SwallowingDifficulty, ClubbingofFingerNails, FrequentCold, DryCough, Snoring]])
#         # Rest of your code for handling the prediction result
#     else:
#         st.warning("Invalid values entered. Please enter valid numerical values for all columns.")

# # ...

    expander = st.expander("Here are some more random values from Test Set")
    
    expander.write(concate_data.head(10))
    