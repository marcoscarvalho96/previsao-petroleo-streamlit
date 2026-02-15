import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error

# =========================
# CONFIG DA PÁGINA
# =========================
st.set_page_config(
    page_title="Previsão do Petróleo",
    layout="wide"
)

st.title("Previsão do Preço do Petróleo (IPEA)")
st.write("""
Aplicação desenvolvida para a disciplina de **Data Analytics**.  
O modelo utiliza **ARIMA otimizado**, respeitando a ordem temporal dos dados.
""")

# =========================
# FUNÇÕES
# =========================
@st.cache_data
def carregar_dados():
    url = "http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view"

    df = pd.read_html(
        url,
        header=0,
        decimal=',',
        thousands='.'
    )[2]

    df.columns = ['data', 'preco_usd']
    df['data'] = pd.to_datetime(df['data'], dayfirst=True)
    df = df.sort_values('data').reset_index(drop=True)

    return df


def calcular_metricas(real, previsto):
    return {
        "MAPE": round(mean_absolute_percentage_error(real, previsto), 4),
        "RMSE": round(np.sqrt(mean_squared_error(real, previsto)), 4),
        "MAE": round(mean_absolute_error(real, previsto), 4)
    }

# =========================
# CARREGAMENTO
# =========================
df = carregar_dados()

st.subheader("Dados Históricos")
st.dataframe(df.tail(10))

fig_hist = px.line(
    df,
    x='data',
    y='preco_usd',
    title="Histórico do Preço do Petróleo (USD)"
)
st.plotly_chart(fig_hist, use_container_width=True)

# =========================
# PREPARAÇÃO
# =========================
df_model = df.rename(columns={'data': 'ds', 'preco_usd': 'y'})

corte = int(len(df_model) * 0.8)
treino = df_model.iloc[:corte]
teste  = df_model.iloc[corte:]

# =========================
# TREINAMENTO ARIMA OTIMIZADO
# =========================
st.subheader("Treinamento do Modelo")

with st.spinner("Otimizando parâmetros do ARIMA..."):
    modelo_auto = auto_arima(
        treino['y'],
        seasonal=False,
        stepwise=True,
        suppress_warnings=True
    )

order_otimizado = modelo_auto.order
st.write(f"**Ordem ARIMA selecionada automaticamente:** {order_otimizado}")

with st.spinner("Treinando ARIMA otimizado..."):
    modelo_arima = ARIMA(
        treino['y'],
        order=order_otimizado
    ).fit()

# =========================
# PREVISÃO NO TESTE
# =========================
previsao_teste = modelo_arima.forecast(steps=len(teste))
teste = teste.copy()
teste['arima_previsto'] = previsao_teste.values

metricas = calcular_metricas(teste['y'], teste['arima_previsto'])

st.subheader("Avaliação do Modelo (Período de Teste)")
st.json(metricas)

# =========================
# PLOT REAL X PREVISTO
# =========================
fig_comp = px.line(
    teste,
    x='ds',
    y=['y', 'arima_previsto'],
    labels={'value': 'Preço (USD)', 'ds': 'Data'},
    title="Preço Real vs Previsão ARIMA (Teste)"
)

st.plotly_chart(fig_comp, use_container_width=True)

# =========================
# PREVISÃO FUTURA
# =========================
st.subheader("Previsão Futura")

periodos = st.slider(
    "Dias para prever:",
    min_value=30,
    max_value=365,
    value=90,
    step=30
)

previsao_futura = modelo_arima.forecast(steps=periodos)

datas_futuras = pd.date_range(
    start=df['data'].max(),
    periods=periodos + 1,
    freq='D'
)[1:]

df_futuro = pd.DataFrame({
    'Data': datas_futuras,
    'Preço Previsto (USD)': previsao_futura.values
})

fig_futuro = px.line(
    df_futuro,
    x='Data',
    y='Preço Previsto (USD)',
    title="Previsão Futura do Preço do Petróleo"
)

st.plotly_chart(fig_futuro, use_container_width=True)
st.dataframe(df_futuro)

# =========================
# DOWNLOAD
# =========================
csv = df_futuro.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Baixar previsão futura (CSV)",
    data=csv,
    file_name='previsao_petroleo_arima.csv',
    mime='text/csv'
)

st.success("Aplicação executada com sucesso!")