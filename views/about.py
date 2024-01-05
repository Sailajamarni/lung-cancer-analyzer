import streamlit as st
import pandas as pd
from PIL import Image, ImageOps

def load_view():    
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """

    st.markdown(hide_st_style, unsafe_allow_html=True)  

    
    tab1, tab2, tab3 , tab4 ,tab5= st.tabs(["Dataset analysis", "Training Data", "Test Data","Algorithms Used",'CNN Based Indentification'])

    with tab1:
        
        st.header("Lung Cancer Dataset")
        data=pd.read_csv("datasets/data.csv")
        st.write(data.head(10))
        code = '''
        Index(['Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Dust Allergy',
       'OccuPational Hazards', 'Genetic Risk', 'chronic Lung Disease',
       'Balanced Diet', 'Obesity', 'Smoking', 'Passive Smoker', 'Chest Pain',
       'Coughing of Blood', 'Fatigue', 'Weight Loss', 'Shortness of Breath',
       'Wheezing', 'Swallowing Difficulty', 'Clubbing of Finger Nails',
       'Frequent Cold', 'Dry Cough', 'Snoring', 'Level'],
      dtype='object')'''
        st.code(code, language='python')
        
        st.header("Pearson Correlation Matrix")
        coors = Image.open("images/coors.png")

        st.image(coors, caption='Pearson Correlation Matrix',width=800)
        st.write("From the above co-relation matrix we did apply a function which picks out values based on their high correlation with a particular attribute which could be dropped to improve Machine Learning Models Performance")
        st.markdown( """
            - The Following Attributed are as follows :-
            """)

        code = '''{'Chest Pain',
 'Coughing of Blood',
 'Dust Allergy',
 'Genetic Risk',
 'OccuPational Hazards',
 'chronic Lung Disease'}'''
        st.code(code, language='python')

    with tab2:
        st.header("Lung Cancer Training Dataset")

        st.subheader("X_Train Data")
        data=pd.read_csv("datasets/train.csv", index_col=0)
        st.write(data)
        code = ''' Index(['Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Balanced Diet',
       'Obesity', 'Smoking', 'Passive Smoker', 'Fatigue', 'Weight Loss',
       'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty',
       'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough', 'Snoring'],
        dtype='object')'''
        st.code(code, language='python')
        data=pd.read_csv("datasets/trainy.csv", index_col=0)
        st.subheader("Y_Train Data")
        st.dataframe(data)

    with tab3:
        st.header("Lung Cancer Training Dataset")

        st.subheader("X_Test Data")
        data=pd.read_csv("datasets/testx.csv", index_col=0)
        st.write(data)
        code = ''' Index(['Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Balanced Diet',
       'Obesity', 'Smoking', 'Passive Smoker', 'Fatigue', 'Weight Loss',
       'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty',
       'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough', 'Snoring'],
        dtype='object')'''
        st.code(code, language='python')
        data=pd.read_csv("datasets/testy.csv", index_col=0)
        st.subheader("Y_Test Data")
        st.dataframe(data)
        
    with tab4:
        st.header("List of Algorithms Used")
        algo = Image.open("images/algo.png")

        st.image(algo, caption='ML Algorithms',width=500)

        st.write("Since this is a Mutlti-Class Classification we have used Algorithms which are maily used for Supervised Learning for the following Problem Statement ")

        st.markdown(
            """
            Supervised Learning Algorithms:
            - Linear Regression
            - Support Vector Machine
            - K-Nearest Neighbours (KNN)
            - Decision Tree Classifier
            """
            )
        
        st.write("The accuracy of all the above algorithms is as follows:- ")
        code = '''The accuracy of the SVM is: 95 %
        The accuracy of the SVM is: 100 %
        The accuracy of Decision Tree is: 100 %
        The accuracy of KNN is: 100 %'''
        st.code(code, language='python')

        st.header("Confusion Matrix")

        col1, col2 = st.columns(2)

        with col1:
            algo = Image.open("images/lg.png")

            st.image(algo, caption='LG Confusion Matrix',width=350)

        with col2:
            algo = Image.open("images/svm.png")

            st.image(algo, caption='SVM Confusion Matrix',width=390)


    with tab5:
        st.header("Convolutional Neural Network Model")
        st.write("Apart from detecting cancer using various parameters in the dataset we can also make out predictions using CT Scan Images by using Convolutional Neural Networks. Link to the image dataset is given below :- ")
        url = "https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images"
        st.write("Check out this [Images Dataset](%s)" % url)

        st.subheader("Approach Followed :- ")
        st.markdown(
            """
            - For training our model we have used the Keras API.
            - We have used 2D Convolution Layer along with consecutive MaxPooling Layers to improve the models performance.
            - Because we are facing a two-class classification problem, i.e. a binary classification problem, we will end the network with a sigmoid activation. The output of the network will be a single scalar between 0 and 1, encoding the probability that the current image is class 1 (as opposed to class 0).
            """
            )
        st.subheader("Model Summary")
        summ = Image.open("images/summary.png")

        st.image(summ, caption='Model Summary',width=700)
        st.subheader("Model Compile ")
        st.write(" You will train our model with the binary_crossentropy loss, because it's a binary classification problem and your final activation is a sigmoid. We will use the rmsprop optimizer with a learning rate of 0.001. During training, you will want to monitor classification accuracy.")
        code = '''from tensorflow.keras.optimizers import RMSprop

        model.compile(optimizer=RMSprop(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics = ['accuracy'])'''
        st.code(code, language='python')

        st.subheader("Fitting Data to the Model")
        st.write(" You will train our model with the binary_crossentropy loss, because it's a binary classification problem and your final activation is a sigmoid. We will use the rmsprop optimizer with a learning rate of 0.001. During training, you will want to monitor classification accuracy.")
        code = '''model.fit(
        train_generator,
        epochs=15,
        validation_data=validation_generator,
        verbose=2
            )'''
        st.code(code, language='python')

        epoc = Image.open("images/epoc.png")

        st.image(epoc, caption='Number of Epocs',width=700)

        st.subheader("Plotting the Traning vs Validation (Accuracy and Loss)")
        col1, col2 = st.columns(2)

        with col1:
            acc = Image.open("images/acc.png")

            st.image(acc, caption='Traning vs Validation Accuracy',width=350)

        with col2:
            loss = Image.open("images/loss.png")

            st.image(loss, caption='Traning vs Validation Loss',width=350)

        st.write("As we can see from the above diagram that our Models performs well on the Training as well as Validation Data")