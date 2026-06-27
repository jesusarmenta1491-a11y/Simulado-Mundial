import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import poisson
import kagglehub
from kagglehub import KaggleDatasetAdapter
import os

# CONFIGURACIÓN
st.set_page_config(page_title="Simulador Mundial 2026", layout="wide")

# --- ENCABEZADO ---
col1, col2 = st.columns([1, 4])

with col1:
    if os.path.exists("copa1.png"):
        st.image("copa.png", use_container_width=True)

with col2:
    st.title("Simulador Inteligente Mundial 2026")
    st.markdown("""
    ### Resumen Estadístico
    Este modelo utiliza **Regresión de Poisson** para estimar probabilidades de goles basadas en el rendimiento histórico.
    """)

if os.path.exists("cancha.jpg"):
    st.image("cancha.jpg", use_container_width=True)

st.markdown("---")

@st.cache_resource
def obtener_modelo():
    df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "martj42/international-football-results-from-1872-to-2017", "results.csv")
    df['date'] = pd.to_datetime(df['date'])
    partidos = df[df['date'].dt.year >= 2000].dropna(subset=['home_score', 'away_score']).copy()
    
    # Datos de respaldo
    datos_2026 = pd.DataFrame([
        {'home_team': 'Portugal', 'away_team': 'DR Congo', 'home_score': 1, 'away_score': 1, 'neutral': True},
        {'home_team': 'Colombia', 'away_team': 'Uzbekistan', 'home_score': 3, 'away_score': 1, 'neutral': True},
        {'home_team': 'England', 'away_team': 'Croatia', 'home_score': 4, 'away_score': 2, 'neutral': True},
        {'home_team': 'Panama', 'away_team': 'Ghana', 'home_score': 1, 'away_score': 0, 'neutral': True},
        {'home_team': 'Argentina', 'away_team': 'Jordan', 'home_score': 2, 'away_score': 0, 'neutral': True}
    ])
    partidos = pd.concat([partidos, pd.concat([datos_2026]*25)], ignore_index=True)
    
    encoder = OneHotEncoder(handle_unknown='ignore').fit(partidos[['home_team', 'away_team']])
    X = np.hstack((encoder.transform(partidos[['home_team', 'away_team']]).toarray(), partidos[['neutral']].astype(int).values))
    
    m_local = PoissonRegressor(alpha=1e-5, max_iter=300).fit(X, partidos['home_score'])
    m_visit = PoissonRegressor(alpha=1e-5, max_iter=300).fit(X, partidos['away_score'])
    return encoder, m_local, m_visit

def cargar_partidos():
    if not os.path.exists("partidos.csv"):
        return {}
    df_jornada = pd.read_csv("partidos.csv")
    return {f"{row['local']} vs {row['visitante']}": (row['local'], row['visitante']) for _, row in df_jornada.iterrows()}

encoder, m_local, m_visit = obtener_modelo()
partidos_disponibles = cargar_partidos()

if partidos_disponibles:
    seleccion = st.selectbox("Selecciona un partido:", list(partidos_disponibles.keys()))
    local, visitante = partidos_disponibles[seleccion]

    if st.button("Calcular Predicción"):
        # Corregido: Crear DataFrame con las columnas correctas para el encoder
        input_data = pd.DataFrame([{'home_team': local, 'away_team': visitante}])
        encoded_data = encoder.transform(input_data[['home_team', 'away_team']]).toarray()
        partido_input = np.hstack((encoded_data, [[1]])) # El 1 asume campo neutral
        
        l_local = m_local.predict(partido_input)[0]
        l_visit = m_visit.predict(partido_input)[0]
        
        st.subheader(f"📊 Pronóstico: {local} vs {visitante}")
        st.info(f"Ataque estimado: {local} ({l_local:.2f}) vs {visitante} ({l_visit:.2f})")
        
        prob_gana, prob_empata, prob_pierde = 0, 0, 0
        resultados = []
        for g_l in range(6):
            for g_v in range(6):
                prob = poisson.pmf(g_l, l_local) * poisson.pmf(g_v, l_visit)
                if g_l > g_v: prob_gana += prob
                elif g_l == g_v: prob_empata += prob
                else: prob_pierde += prob
                resultados.append({"Marcador": f"{g_l} - {g_v}", "Probabilidad": prob})
        
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Victoria {local}", f"{prob_gana*100:.1f}%")
        c2.metric("Empate", f"{prob_empata*100:.1f}%")
        c3.metric(f"Victoria {visitante}", f"{prob_pierde*100:.1f}%")
        
        df_resultados = pd.DataFrame(resultados)
        df_resultados["Probabilidad"] = df_resultados["Probabilidad"].apply(lambda x: f"{x*100:.2f}%")
        st.table(df_resultados.sort_values(by="Probabilidad", ascending=False).head(10))
else:
    st.warning("No se pudo cargar 'partidos.csv'. Asegúrate de que existe.")
