import streamlit as st
import pandas as pd
import os
from sklearn.linear_model import LinearRegression

csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'pizzas.csv')
df = pd.read_csv(csv_path)

modelo = LinearRegression()
x = df[['diametro']]
y = df[['preco']]

modelo.fit(x, y)

st.title('Prevendo o valor de uma pizza')
# st.divider()
diametro = st.number_input('Digite o diâmetro da pizza:')

if diametro:
    preco_previsto = modelo.predict([[diametro]])[0][0]
    st.write(f'O valor da pizza com diâmetro de {diametro:.2f} cm é de R$ {preco_previsto:.2f}')
    st.balloons()
