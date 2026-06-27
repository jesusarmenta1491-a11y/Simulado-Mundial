import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import poisson
import kagglehub
from kagglehub import KaggleDatasetAdapter

st.set_page_config(page_title="Simulador Mundial 2026", layout="wide")
st.title("⚽ Simulador Inteligente Mundial 2026")

# --- MODELO ---
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

# --- LECTURA DE PARTIDOS DESDE CSV ---
def cargar_partidos():
    try:
        df_jornada = pd.read_csv("partidos.csv")
        return {f"{row['local']} vs {row['visitante']}": (row['local'], row['visitante']) 
                for _, row in df_jornada.iterrows()}
    except FileNotFoundError:
        st.error("No se encontró 'partidos.csv'. Asegúrate de que esté en la carpeta.")
        return {}
    except KeyError:
        st.error("Error en el CSV. Asegúrate de que la primera fila sea: local,visitante")
        return {}

# --- LÓGICA DE ANÁLISIS ---
def generar_analisis(l_local, l_visit, local, visitante):
    diff = l_local - l_visit
    if abs(diff) < 0.3:
        return f"Encuentro muy equilibrado entre {local} y {visitante}."
    elif diff > 0:
        return f"Las estadísticas favorecen a {local} con un ataque estimado de {l_local:.2f} goles."
    else:
        return f"El equipo de {visitante} llega con una proyección superior de {l_visit:.2f} goles."

# --- INTERFAZ ---
encoder, m_local, m_visit = obtener_modelo()
partidos_disponibles = cargar_partidos()

if partidos_disponibles:
    seleccion = st.selectbox("Selecciona un partido:", list(partidos_disponibles.keys()))
    local, visitante = partidos_disponibles[seleccion]

    if st.button("Calcular Predicción"):
        partido = np.hstack((encoder.transform(pd.DataFrame([{'home_team': local, 'away_team': visitante}]))[0].toarray(), [[1]]))
        l_local = m_local.predict(partido)[0]
        l_visit = m_visit.predict(partido)[0]
        
        st.subheader(f"📊 Pronóstico: {local} vs {visitante}")
        st.info(generar_analisis(l_local, l_visit, local, visitante))
        
        # --- CÁLCULO DE PROBABILIDADES ---
        prob_gana, prob_empata, prob_pierde = 0, 0, 0
        resultados = []
        for g_l in range(6):
            for g_v in range(6):
                prob = poisson.pmf(g_l, l_local) * poisson.pmf(g_v, l_visit)
                if g_l > g_v: prob_gana += prob
                elif g_l == g_v: prob_empata += prob
                else: prob_pierde += prob
                resultados.append({"Marcador": f"{local} {g_l} - {g_v} {visitante}", "Probabilidad (%)": f"{prob*100:.2f}%"})
        
        # --- MOSTRAR MÉTRICAS ---
        col1, col2, col3 = st.columns(3)
        col1.metric(f"Victoria {local}", f"{prob_gana*100:.1f}%")
        col2.metric("Empate", f"{prob_empata*100:.1f}%")
        col3.metric(f"Victoria {visitante}", f"{prob_pierde*100:.1f}%")
        
        st.write("---")
        st.table(pd.DataFrame(resultados).sort_values(by="Probabilidad (%)", ascending=False).head(10))