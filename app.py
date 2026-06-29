import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import poisson
import kagglehub
from kagglehub import KaggleDatasetAdapter

st.set_page_config(page_title="Simulador Mundial 2026 - Eliminatorias", layout="wide")
st.title("🏆 Simulador Inteligente: Fase Eliminatoria")

# --- MODELO ---
@st.cache_resource
def obtener_modelo():
    # Carga de datos históricos
    df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "martj42/international-football-results-from-1872-to-2017", "results.csv")
    df['date'] = pd.to_datetime(df['date'])
    partidos = df[df['date'].dt.year >= 2000].dropna(subset=['home_score', 'away_score']).copy()
    
    # Inyección de actualidad (Fase de grupos 2026)
    datos_2026 = pd.DataFrame([
        {'home_team': 'Portugal', 'away_team': 'DR Congo', 'home_score': 1, 'away_score': 1, 'neutral': True},
        {'home_team': 'Colombia', 'away_team': 'Uzbekistan', 'home_score': 3, 'away_score': 1, 'neutral': True},
        {'home_team': 'Portugal', 'away_team': 'Uzbekistan', 'home_score': 5, 'away_score': 0, 'neutral': True},
        {'home_team': 'Colombia', 'away_team': 'DR Congo', 'home_score': 1, 'away_score': 0, 'neutral': True},
        {'home_team': 'England', 'away_team': 'Croatia', 'home_score': 4, 'away_score': 2, 'neutral': True},
        {'home_team': 'Ghana', 'away_team': 'Panama', 'home_score': 1, 'away_score': 0, 'neutral': True},
        {'home_team': 'England', 'away_team': 'Ghana', 'home_score': 0, 'away_score': 0, 'neutral': True},
        {'home_team': 'Panama', 'away_team': 'Croatia', 'home_score': 0, 'away_score': 1, 'neutral': True},
        {'home_team': 'Argentina', 'away_team': 'Algeria', 'home_score': 3, 'away_score': 0, 'neutral': True},
        {'home_team': 'Austria', 'away_team': 'Jordan', 'home_score': 3, 'away_score': 1, 'neutral': True},
        {'home_team': 'Argentina', 'away_team': 'Austria', 'home_score': 2, 'away_score': 0, 'neutral': True},
        {'home_team': 'Jordan', 'away_team': 'Algeria', 'home_score': 1, 'away_score': 2, 'neutral': True}
    ])
    
    # Aplicamos el sobrepeso a la actualidad
    peso_actualidad = 25
    datos_2026_pesados = pd.concat([datos_2026] * peso_actualidad, ignore_index=True)
    partidos = pd.concat([partidos, datos_2026_pesados], ignore_index=True)
    
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
        st.error("No se encontró 'partidos.csv'. Asegúrate de crear este archivo para los cruces eliminatorios.")
        return {}
    except KeyError:
        st.error("Error en el CSV. Asegúrate de que la primera fila sea exactamente: local,visitante")
        return {}

# --- LÓGICA DE ANÁLISIS ---
def generar_analisis_eliminatoria(prob_gana, prob_empata, prob_pierde, local, visitante):
    if prob_empata > 0.30:
        return f"Alto riesgo de Prórroga. Las probabilidades de que el partido se extienda más allá de los 90 minutos son significativas."
    elif prob_gana > prob_pierde:
        return f"{local} es favorito para clasificar a la siguiente ronda dentro de los 90 minutos reglamentarios."
    else:
        return f"{visitante} llega con mejores proyecciones para asegurar su pase sin necesidad de tiempo extra."

# --- INTERFAZ ---
encoder, m_local, m_visit = obtener_modelo()
partidos_disponibles = cargar_partidos()

if partidos_disponibles:
    st.markdown("### 🏟️ Cruces de Eliminatoria")
    seleccion = st.selectbox("Selecciona el partido a simular:", list(partidos_disponibles.keys()))
    local, visitante = partidos_disponibles[seleccion]

    if st.button("Ejecutar Simulación de 90 Minutos"):
        partido = np.hstack((encoder.transform(pd.DataFrame([{'home_team': local, 'away_team': visitante}]))[0].toarray(), [[1]]))
        l_local = m_local.predict(partido)[0]
        l_visit = m_visit.predict(partido)[0]
        
        # --- CÁLCULO DE PROBABILIDADES ---
        prob_gana, prob_empata, prob_pierde = 0, 0, 0
        resultados = []
        for g_l in range(6):
            for g_v in range(6):
                prob = poisson.pmf(g_l, l_local) * poisson.pmf(g_v, l_visit)
                if g_l > g_v: prob_gana += prob
                elif g_l == g_v: prob_empata += prob
                else: prob_pierde += prob
                resultados.append({"Marcador Exacto (90 min)": f"{local} {g_l} - {g_v} {visitante}", "Probabilidad (%)": f"{prob*100:.2f}%"})
        
        st.write("---")
        st.info(generar_analisis_eliminatoria(prob_gana, prob_empata, prob_pierde, local, visitante))
        
        # --- MOSTRAR MÉTRICAS ADAPTADAS A ELIMINATORIAS ---
        col1, col2, col3 = st.columns(3)
        col1.metric(f"Clasifica {local} (Directo)", f"{prob_gana*100:.1f}%")
        col2.metric("Alargue / Penales", f"{prob_empata*100:.1f}%")
        col3.metric(f"Clasifica {visitante} (Directo)", f"{prob_pierde*100:.1f}%")
        
        st.write("---")
        st.markdown("#### 🎯 Proyección de Marcadores (Fin del tiempo regular)")
        st.table(pd.DataFrame(resultados).sort_values(by="Probabilidad (%)", ascending=False).head(8))
