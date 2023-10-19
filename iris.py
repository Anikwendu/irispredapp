import streamlit as st
import pandas as pd
from PIL import Image
import pickle
import sklearn
my_iris_model = pickle.load(open("C:\\Users\\Amarachi Uzochukwu\\Desktop\\iris_model.pkl",'rb'))
st.title('IRIS FLOWER PREDICTION APP')
img = Image.open('C:\\Users\\Amarachi Uzochukwu\\Downloads\\pexels-kaitlyn-epperson-8444498.jpg')
st.image(img, width=350)
def user_report():
    sepal_length = st.sidebar.slider('sepal length', 1.0, 10.0, 0.1)
    sepal_width = st.sidebar.slider('sepal width', 1.0, 10.0, 0.1)
    petal_length = st.sidebar.slider('petal length', 1.0, 10.0, 0.1)
    petal_width = st.sidebar.slider('petal width', 1.0, 10.0, 0.1)

    report_data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    data = pd.DataFrame(report_data, index=[0])
    return data
user_data = user_report()
st.write(user_data)
prediction = my_iris_model.predict(user_data)
if (prediction==0):
    img = Image.open("C:\\Users\\Amarachi Uzochukwu\\Downloads\\pexels-aaron-burden-2471455.jpg")
    st.success('setosa')
elif (prediction==1):
    img = Image.open("C:\\Users\\Amarachi Uzochukwu\\Downloads\\pexels-lutz-rolke-8046199.jpg")
    st.success('versicolor')
else:
    img = Image.open("C:\\Users\\Amarachi Uzochukwu\\Downloads\\pexels-harvey-alston-5902401")
    st.success('virginica')