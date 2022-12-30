#librerias
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


import category_encoders as catEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model  import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from PIL import Image

#esquema del dashboard
header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()


with st.sidebar:
    st.title('Proyecto con Streamlit')
    image1 = Image.open(r'data/logo.png')
    st.image(image1)
    st.markdown('Utilizacion de Streamlit para implementacion de modelos de machine learning en python')
    
    st.markdown('Creado por:  **Herrera H. Federico**')
    st.markdown('[Linkedin](https://www.linkedin.com/in/h%C3%A9ctor-federico-herrera/)', unsafe_allow_html=True)
    st.markdown('[Github](https://github.com/f3derico1991/f3derico1991)', unsafe_allow_html=True)

with header:
    st.title('Analisis y Predicción de Prestamos')

    st.header('Descripcion de los datos')
    st.markdown('''Entre todas las industrias, el dominio de seguros tiene el mayor uso de métodos de análisis y ciencia de datos. 
            Este conjunto de datos le proporcionaría suficiente información para trabajar con conjuntos de datos de compañías de seguros, 
            qué desafíos se enfrentan, qué estrategias se utilizan, qué variables influyen en el resultado, etc. Este es un problema de clasificación.''') 



with dataset:
    st.header('Data-Set')
    st.markdown('[Kaggle descarga dataset](https://www.kaggle.com/datasets/ninzaami/loan-predication)')
    st.markdown('''Los datos tienen 615 filas y 13 columnas.
            *Problema*: La empresa desea automatizar el proceso de elegibilidad del préstamo (en tiempo real) 
            en función de los detalles del cliente proporcionados al completar el formulario de solicitud en línea. Estos detalles son Género, 
            Estado Civil, Educación, Número de Dependientes, Ingresos, Monto del Préstamo, Historial de Crédito y otros. 
            Para automatizar este proceso, han dado un problema para identificar los segmentos de clientes, aquellos son elegibles para el monto del préstamo para que puedan dirigirse 
            específicamente a estos clientes. Aquí han proporcionado un conjunto de datos parcial.''')

    df = pd.read_csv(r'https://docs.google.com/spreadsheets/d/e/2PACX-1vRv2ZjjEw24DPfYUzW21PoOVOrdfK0NH5FFvJgJtG3KZA46hUB5GcPBjulNM23Mfut5MqW9BcyXQzH8/pub?gid=302117549&single=true&output=csv')
    st.table(df.head(10))

    

with features:
    st.header('Analisis de variables categoricas')

    variables_caterogicas = ['Gender',"Married","Property_Area","Education"]

    st.markdown('##### *Graficos de variabes categoricas*')
    for i in variables_caterogicas:
        with st.expander(i):
            grafico = px.histogram(df, x=i)
            st.text(f"Cantidad de clientes según {i}")
            st.plotly_chart(grafico)


with model_training:
    st.header('Modelos de Clasificasion')
    #tratamiento de datos
    categ_data = ['Gender',"Married","Property_Area",'Education','Self_Employed']
    BE = catEncoder.BinaryEncoder(cols=categ_data)
    df['Dependents'] = df['Dependents'].replace(to_replace='3+',value=0)
    df = df.dropna()
    df = BE.fit_transform(df)
    y = df['Loan_Status'].replace(to_replace=['Y','N'],value=[1,0])
    X = df.drop(['Loan_ID','Loan_Status'],axis=1)

    sel_col,disp_col = st.columns(2)
    seleccion = sel_col.selectbox('Selecciona el modelo a entrenar: ',['Naives_Bayes','Arbol_desicion'],index=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    if seleccion == 'Naives Bayes':
        modelo = GaussianNB()
    # elif seleccion == 'Regresion_logistica':
    #     modelo = LogisticRegression()
    else:
        modelo = DecisionTreeClassifier()

    modelo_train = modelo.fit(X_train, y_train)
    y_pred = modelo_train.predict(X_test)

    disp_col.markdown('##### Accuracy del modelo es: ')
    disp_col.write(accuracy_score(y_pred,y_test))


    matrix = confusion_matrix(y_pred, y_test)
    fig = plt.figure(figsize=(10, 8))
    grafico = sns.heatmap(matrix, annot=True)
    st.subheader('Matriz de Confucion')
    st.pyplot(fig)
