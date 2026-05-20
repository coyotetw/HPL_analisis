import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re


# ============================================================
# CONFIGURACIÓN GENERAL DE LA APP
# ============================================================

st.set_page_config(
    page_title="Sprinklr Core - Airline Monitoring",
    layout="wide"
)

# ============================================================
# ESTILO AL DASHBOARD
# ============================================================
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }

    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 14px;
        border-radius: 14px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.06);
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: #374151;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.7rem;
        font-weight: 700;
    }

    .dashboard-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 12px;
    }

    .small-caption {
        color: #6b7280;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 1. CARGA Y PREPROCESAMIENTO DEL DATASET
# ============================================================

@st.cache_data
def descargar_dataset_real():
    archivo_local = "Tweets.csv"

    try:
        df_raw = pd.read_csv(archivo_local)

        # Seleccionamos las columnas clave del dataset
        df_clean = df_raw[['text', 'airline_sentiment', 'negativereason', 'airline']].copy()
        df_clean.columns = ['Text', 'Sentiment', 'Topic', 'Airline']

        # Usuario simulado para la visualización
        df_clean['User'] = "@passenger_ir"

        # Limpieza de nulos en tópicos
        df_clean['Topic'] = df_clean['Topic'].fillna("General / Other")

        # Traducción de tópicos frecuentes
        mapa_topicos = {
            'Customer Service Issue': 'Atención al Cliente',
            'Late Flight': 'Vuelo Demorado',
            'Cancelled Flight': 'Vuelo Cancelado',
            'Lost Luggage': 'Equipaje Perdido',
            'Bad Flight': 'Experiencia de Vuelo Mala',
            'Flight Booking Problems': 'Problemas de Reserva',
            'Flight Attendant Complaints': 'Quejas del Personal',
            'Long Lines': 'Filas Largas',
            'Damaged Luggage': 'Equipaje Dañado'
        }

        df_clean['Topic'] = df_clean['Topic'].replace(mapa_topicos)

        # Normalización de sentimiento
        df_clean['Sentiment'] = df_clean['Sentiment'].astype(str).str.strip().str.lower()

        mapa_sentimientos = {
            'positive': 'Positivo',
            'negative': 'Negativo',
            'neutral': 'Neutro'
        }

        df_clean['Sentiment'] = df_clean['Sentiment'].map(mapa_sentimientos)

        # Coordenadas simuladas para ubicar problemas por aerolínea / hub principal
        # Coordenadas simuladas para ubicar problemas por aerolínea / hubs principales
        # Nota: algunas ubicaciones internacionales son agregadas para simular monitoreo global.
        mapa_ubicaciones = {
            # América del Norte
            "United": {
                "Ciudad": "Chicago",
                "Aeropuerto": "ORD",
                "Region": "América del Norte",
                "Latitud": 41.9742,
                "Longitud": -87.9073
            },
            "US Airways": {
                "Ciudad": "Charlotte",
                "Aeropuerto": "CLT",
                "Region": "América del Norte",
                "Latitud": 35.2144,
                "Longitud": -80.9473
            },
            "American": {
                "Ciudad": "Dallas",
                "Aeropuerto": "DFW",
                "Region": "América del Norte",
                "Latitud": 32.8998,
                "Longitud": -97.0403
            },
            "Southwest": {
                "Ciudad": "Dallas",
                "Aeropuerto": "DAL",
                "Region": "América del Norte",
                "Latitud": 32.8471,
                "Longitud": -96.8518
            },
            "Delta": {
                "Ciudad": "Atlanta",
                "Aeropuerto": "ATL",
                "Region": "América del Norte",
                "Latitud": 33.6407,
                "Longitud": -84.4277
            },
            "JetBlue": {
                "Ciudad": "New York",
                "Aeropuerto": "JFK",
                "Region": "América del Norte",
                "Latitud": 40.6413,
                "Longitud": -73.7781
            },
            "Virgin America": {
                "Ciudad": "San Francisco",
                "Aeropuerto": "SFO",
                "Region": "América del Norte",
                "Latitud": 37.6213,
                "Longitud": -122.3790
            }
        }

        df_clean["Ciudad"] = df_clean["Airline"].map(
            lambda x: mapa_ubicaciones.get(x, {}).get("Ciudad", "Sin ubicación")
        )

        df_clean["Aeropuerto"] = df_clean["Airline"].map(
            lambda x: mapa_ubicaciones.get(x, {}).get("Aeropuerto", "N/D")
        )

        df_clean["Region"] = df_clean["Airline"].map(
            lambda x: mapa_ubicaciones.get(x, {}).get("Region", "Sin región")
        )

        df_clean["Latitud"] = df_clean["Airline"].map(
            lambda x: mapa_ubicaciones.get(x, {}).get("Latitud", None)
        )

        df_clean["Longitud"] = df_clean["Airline"].map(
            lambda x: mapa_ubicaciones.get(x, {}).get("Longitud", None)
        )

        df_clean["Ciudad"] = df_clean["Airline"].map(
            lambda x: mapa_ubicaciones.get(x, {}).get("Ciudad", "Sin ubicación")
        )

        df_clean["Aeropuerto"] = df_clean["Airline"].map(
            lambda x: mapa_ubicaciones.get(x, {}).get("Aeropuerto", "N/D")
        )

        df_clean["Latitud"] = df_clean["Airline"].map(
            lambda x: mapa_ubicaciones.get(x, {}).get("Latitud", None)
        )

        df_clean["Longitud"] = df_clean["Airline"].map(
            lambda x: mapa_ubicaciones.get(x, {}).get("Longitud", None)
        )



        # Eliminamos posibles filas sin sentimiento válido
        df_clean = df_clean.dropna(subset=['Sentiment'])

        # Fecha simulada para poder mostrar evolución temporal
        df_clean['Fecha'] = pd.date_range(
            end=pd.Timestamp.today(),
            periods=len(df_clean),
            freq="min"
        )

        return df_clean

    except FileNotFoundError:
        st.error(
            f"No se encontró el archivo '{archivo_local}' en la carpeta del proyecto. "
            "Asegurate de que esté en la misma carpeta que app.py."
        )

        return pd.DataFrame(
            columns=[
                'Text', 'Sentiment', 'Topic', 'User', 'Airline',
                'Fecha', 'Ciudad', 'Aeropuerto', 'Region', 'Latitud', 'Longitud'
            ]
        )


# ============================================================
# 2. FUNCIONES AUXILIARES
# ============================================================

def agregar_puntos_globales_simulados(df):
    """
    Agrega puntos internacionales simulados para que el mapa represente
    un monitoreo global. No son ubicaciones reales de los tweets.
    Genera muestras negativas, positivas y neutras para permitir flujo dinámico.
    """

    if df.empty:
        return df

    puntos_globales = [
        {
            "Airline": "British Airways",
            "Ciudad": "Londres",
            "Aeropuerto": "LHR",
            "Region": "Europa",
            "Latitud": 51.4700,
            "Longitud": -0.4543
        },
        {
            "Airline": "Air France",
            "Ciudad": "París",
            "Aeropuerto": "CDG",
            "Region": "Europa",
            "Latitud": 49.0097,
            "Longitud": 2.5479
        },
        {
            "Airline": "Lufthansa",
            "Ciudad": "Frankfurt",
            "Aeropuerto": "FRA",
            "Region": "Europa",
            "Latitud": 50.0379,
            "Longitud": 8.5622
        },
        {
            "Airline": "Emirates",
            "Ciudad": "Dubái",
            "Aeropuerto": "DXB",
            "Region": "Asia / Medio Oriente",
            "Latitud": 25.2532,
            "Longitud": 55.3657
        },
        {
            "Airline": "Qatar Airways",
            "Ciudad": "Doha",
            "Aeropuerto": "DOH",
            "Region": "Asia / Medio Oriente",
            "Latitud": 25.2731,
            "Longitud": 51.6081
        },
        {
            "Airline": "Singapore Airlines",
            "Ciudad": "Singapur",
            "Aeropuerto": "SIN",
            "Region": "Asia",
            "Latitud": 1.3644,
            "Longitud": 103.9915
        },
        {
            "Airline": "LATAM",
            "Ciudad": "Santiago de Chile",
            "Aeropuerto": "SCL",
            "Region": "América del Sur",
            "Latitud": -33.3928,
            "Longitud": -70.7858
        },
        {
            "Airline": "Aerolíneas Argentinas",
            "Ciudad": "Buenos Aires",
            "Aeropuerto": "EZE",
            "Region": "América del Sur",
            "Latitud": -34.8222,
            "Longitud": -58.5358
        },
        {
            "Airline": "Qantas",
            "Ciudad": "Sídney",
            "Aeropuerto": "SYD",
            "Region": "Oceanía",
            "Latitud": -33.9399,
            "Longitud": 151.1753
        },
        {
            "Airline": "Air New Zealand",
            "Ciudad": "Auckland",
            "Aeropuerto": "AKL",
            "Region": "Oceanía",
            "Latitud": -37.0082,
            "Longitud": 174.7850
        }
    ]

    muestras = []

    for i, punto in enumerate(puntos_globales):
        for sentimiento in ["Negativo", "Positivo", "Neutro"]:
            df_sentimiento = df[df["Sentiment"] == sentimiento].copy()

            if df_sentimiento.empty:
                continue

            muestra = df_sentimiento.sample(
                n=min(25, len(df_sentimiento)),
                random_state=1000 + i
            ).copy()

            muestra["Airline"] = punto["Airline"]
            muestra["Ciudad"] = punto["Ciudad"]
            muestra["Aeropuerto"] = punto["Aeropuerto"]
            muestra["Region"] = punto["Region"]
            muestra["Latitud"] = punto["Latitud"]
            muestra["Longitud"] = punto["Longitud"]

            muestras.append(muestra)

    if not muestras:
        return df

    df_global = pd.concat(muestras, ignore_index=True)

    return pd.concat([df, df_global], ignore_index=True)

def mostrar_recomendacion_dinamica(modo_dashboard, riesgo_crisis, top_topic, positivos, negativos, total_menciones):
    st.write("### Recomendación Estratégica del Sistema")

    porcentaje_positivo = round((positivos / total_menciones) * 100, 2) if total_menciones > 0 else 0
    porcentaje_negativo = round((negativos / total_menciones) * 100, 2) if total_menciones > 0 else 0

    if modo_dashboard == "crisis":
        if riesgo_crisis in ["CRÍTICO", "ALTO"]:
            st.markdown(f"""
            **Acción recomendada:** activar protocolo de crisis reputacional.

            - Priorizar respuestas públicas a usuarios afectados.
            - Escalar los casos críticos al área correspondiente.
            - Preparar un comunicado oficial si el problema se concentra en un mismo tópico.
            - Reforzar la atención al cliente durante las próximas horas.
            - Monitorear la evolución de menciones negativas.

            **Foco principal detectado:** `{top_topic if top_topic else "No identificado"}`
            """)
        else:
            st.markdown("""
            **Acción recomendada:** mantener monitoreo preventivo.

            - Revisar reclamos recientes.
            - Observar si aumenta el volumen de menciones negativas.
            - Preparar respuestas para casos sensibles.
            """)

    elif modo_dashboard == "campaña":
        st.markdown(f"""
        **Acción recomendada:** capitalizar el impacto positivo de la campaña.

        - Identificar usuarios satisfechos para acciones de fidelización.
        - Reutilizar menciones positivas como insumos para comunicación institucional.
        - Detectar qué aerolíneas o regiones concentran mejor respuesta.
        - Mantener seguimiento por si aparecen reclamos asociados a la campaña.

        **Porcentaje de menciones positivas:** `{porcentaje_positivo}%`
        """)

    else:
        st.markdown(f"""
        **Acción recomendada:** mantener monitoreo general.

        - Observar equilibrio entre menciones positivas y negativas.
        - Detectar cambios bruscos de tendencia.
        - Revisar oportunidades de fidelización y posibles focos de crisis.

        **Menciones positivas:** `{porcentaje_positivo}%`  
        **Menciones negativas:** `{porcentaje_negativo}%`
        """)

def calcular_metricas(df):
    total = len(df)
    positivos = len(df[df["Sentiment"] == "Positivo"])
    negativos = len(df[df["Sentiment"] == "Negativo"])
    neutros = len(df[df["Sentiment"] == "Neutro"])

    nsi = int(((positivos - negativos) / total) * 100) if total > 0 else 0
    porcentaje_negativo = round((negativos / total) * 100, 2) if total > 0 else 0

    return total, positivos, negativos, neutros, nsi, porcentaje_negativo


def calcular_riesgo_crisis(total_menciones, porcentaje_negativo, nsi):
    """
    Calcula un nivel de riesgo combinando volumen, negatividad y Net Sentiment Index.
    """

    if total_menciones == 0:
        return "SIN DATOS"

    if porcentaje_negativo >= 60 and total_menciones >= 100:
        return "CRÍTICO"
    elif porcentaje_negativo >= 45 or nsi <= -40:
        return "ALTO"
    elif porcentaje_negativo >= 25 or nsi <= -20:
        return "MEDIO"
    else:
        return "BAJO"


def mostrar_semaforo(riesgo_crisis):
    if riesgo_crisis == "CRÍTICO":
        st.error("🚨 Riesgo crítico de crisis reputacional. Se recomienda activar protocolo de respuesta inmediata.")
    elif riesgo_crisis == "ALTO":
        st.warning("⚠️ Riesgo alto. Aumentan las menciones negativas y los reclamos.")
    elif riesgo_crisis == "MEDIO":
        st.info("🟡 Riesgo medio. Conviene monitorear la evolución de las menciones.")
    elif riesgo_crisis == "BAJO":
        st.success("🟢 Situación estable. No se detectan señales críticas.")
    else:
        st.info("No hay datos suficientes para calcular el riesgo.")


def color_sentiment_row(val):
    if val == "Positivo":
        return 'background-color: rgba(46, 204, 113, 0.15)'
    elif val == "Negativo":
        return 'background-color: rgba(231, 76, 60, 0.15)'
    elif val == "Neutro":
        return 'background-color: rgba(241, 196, 15, 0.08)'
    return ''


def generar_ranking_aerolineas(df):
    if df.empty:
        return pd.DataFrame()

    df_ranking = df.groupby("Airline").agg(
        Total=("Sentiment", "count"),
        Negativos=("Sentiment", lambda x: (x == "Negativo").sum()),
        Positivos=("Sentiment", lambda x: (x == "Positivo").sum()),
        Neutros=("Sentiment", lambda x: (x == "Neutro").sum())
    ).reset_index()

    df_ranking["% Negativo"] = round((df_ranking["Negativos"] / df_ranking["Total"]) * 100, 2)
    df_ranking["Net Sentiment"] = round(
        ((df_ranking["Positivos"] - df_ranking["Negativos"]) / df_ranking["Total"]) * 100,
        2
    )

    df_ranking = df_ranking.sort_values("% Negativo", ascending=False)

    return df_ranking


def obtener_topico_dominante(df):
    df_neg_topics = df[
        (df["Sentiment"] == "Negativo") &
        (df["Topic"] != "General / Other")
    ]

    if df_neg_topics.empty:
        return None, 0

    top_topic = df_neg_topics["Topic"].value_counts().idxmax()
    top_topic_count = df_neg_topics["Topic"].value_counts().max()

    return top_topic, top_topic_count


def mostrar_recomendacion_estrategica(riesgo_crisis, top_topic):
    st.write("### Recomendación Estratégica del Sistema")

    if riesgo_crisis == "CRÍTICO":
        st.markdown(f"""
        **Acción recomendada:** activar protocolo de crisis reputacional.

        - Priorizar respuestas públicas a usuarios afectados.
        - Escalar los casos críticos al área correspondiente.
        - Preparar un comunicado oficial si el problema se concentra en un mismo tópico.
        - Reforzar la atención al cliente durante las próximas horas.
        - Monitorear la evolución de las menciones en intervalos cortos.

        **Foco principal detectado:** `{top_topic if top_topic else "No identificado"}`
        """)

    elif riesgo_crisis == "ALTO":
        st.markdown(f"""
        **Acción recomendada:** reforzar atención y prevención.

        - Responder reclamos negativos recientes.
        - Identificar si el problema corresponde a demoras, cancelaciones, equipaje o atención.
        - Preparar mensajes preventivos para redes sociales.
        - Observar si el tópico dominante se mantiene o se expande.

        **Foco principal detectado:** `{top_topic if top_topic else "No identificado"}`
        """)

    elif riesgo_crisis == "MEDIO":
        st.markdown("""
        **Acción recomendada:** continuar monitoreo activo.

        - Revisar tópicos emergentes.
        - Observar si aumenta el volumen de menciones negativas.
        - Detectar usuarios o reclamos repetidos.
        """)

    elif riesgo_crisis == "BAJO":
        st.markdown("""
        **Acción recomendada:** mantener seguimiento regular.

        - Aprovechar menciones positivas para campañas de fidelización.
        - Identificar usuarios satisfechos.
        - Continuar monitoreando posibles cambios de tendencia.
        """)

    else:
        st.markdown("""
        No hay datos suficientes para generar una recomendación.
        """)


def generar_terminos_frecuentes(df, sentimiento="Negativo"):
    textos = " ".join(
        df[df["Sentiment"] == sentimiento]["Text"].astype(str)
    )

    palabras = re.findall(r'\b[a-zA-Z]{4,}\b', textos.lower())

    stopwords = {
        "the", "and", "for", "you", "your", "are", "was", "were", "with",
        "this", "that", "have", "has", "had", "from", "they", "them",
        "then", "than", "there", "their", "will", "would", "could", "should",
        "can", "cant", "dont", "didnt", "not", "but", "all", "our",
        "out", "get", "got", "now", "why", "how", "too", "just",
        "what", "when", "where", "about", "because", "been", "again",
        "very", "more", "only", "any", "yes", "amp", "http", "https",
        "flight", "flights", "airline", "airlines", "plane",
        "united", "unitedairlines", "usairways", "americanair",
        "southwestair", "jetblue", "virginamerica", "delta", "southwest",
        "im", "ive"
    }

    palabras_filtradas = [p for p in palabras if p not in stopwords]

    conteo = Counter(palabras_filtradas).most_common(15)

    return pd.DataFrame(conteo, columns=["Palabra", "Frecuencia"])


# ============================================================
# 3. INICIALIZACIÓN DE DATOS
# ============================================================

with st.spinner("Leyendo y preprocesando dataset real de Twitter de forma local..."):
    df_master = descargar_dataset_real()
    df_master = agregar_puntos_globales_simulados(df_master)

if df_master.empty:
    st.stop()

# Inicializamos dataset de trabajo en sesión
if 'dataset' not in st.session_state:
    cantidad_muestra = min(1500, len(df_master))
    st.session_state.dataset = df_master.sample(n=cantidad_muestra, random_state=42).copy()
    st.session_state.estado = "Monitoreo Histórico Completo (Datos Reales)"


# ============================================================
# 4. INTERFAZ PRINCIPAL
# ============================================================

st.title("📊 Sprinklr AI - Monitorización de redes sociales para múltiples aerolíneas")

aerolinea_titulo = "Global Airlines Pool"

st.sidebar.header("Centro de Control e Ingesta")

lista_aerolineas = ["Todas"] + sorted(list(df_master['Airline'].dropna().unique()))

aerolinea_seleccionada = st.sidebar.selectbox(
    "✈️ Seleccionar Aerolínea a Monitorear:",
    lista_aerolineas
)

if aerolinea_seleccionada != "Todas":
    aerolinea_titulo = aerolinea_seleccionada

st.subheader(f"Cliente: {aerolinea_titulo} | Proceso de análisis de PLN en tiempo real")
st.markdown("---")

st.sidebar.markdown("**Simulación de Ingesta (NLP Engine):**")


# ============================================================
# 5. BOTONES DE SIMULACIÓN
# ============================================================

if st.sidebar.button("Resetear Datos", use_container_width=True):
    cantidad_muestra = min(1500, len(df_master))
    st.session_state.dataset = df_master.sample(n=cantidad_muestra, random_state=42).copy()
    st.session_state.estado = "Monitoreo Histórico Completo (Datos Reales)"
    st.toast("Datos reseteados al estado histórico real.")


if st.sidebar.button("Inyectar Flujo Crisis (Negativo)", use_container_width=True):
    df_negativos = df_master[df_master['Sentiment'] == 'Negativo']

    if aerolinea_seleccionada != "Todas":
        df_negativos = df_negativos[df_negativos['Airline'] == aerolinea_seleccionada]

    if not df_negativos.empty:
        tweets_negativos = df_negativos.sample(min(150, len(df_negativos)), random_state=42)
        st.session_state.dataset = pd.concat(
            [st.session_state.dataset, tweets_negativos],
            ignore_index=True
        )
        st.session_state.estado = f"Crisis Activa Detectada en {aerolinea_seleccionada if aerolinea_seleccionada != 'Todas' else 'el Pool'}"
        st.toast("¡Alerta de anomalía! Alto volumen de quejas reales ingresando.")
    else:
        st.sidebar.error("No se encontraron suficientes tweets negativos.")


if st.sidebar.button("Inyectar Campaña (Positivo)", use_container_width=True):
    df_positivos = df_master[df_master['Sentiment'] == 'Positivo']

    if aerolinea_seleccionada != "Todas":
        df_positivos = df_positivos[df_positivos['Airline'] == aerolinea_seleccionada]

    if not df_positivos.empty:
        tweets_positivos = df_positivos.sample(min(150, len(df_positivos)), random_state=42)
        st.session_state.dataset = pd.concat(
            [st.session_state.dataset, tweets_positivos],
            ignore_index=True
        )
        st.session_state.estado = "Mitigación Activa / Impacto de Campaña"
        st.toast("Procesando menciones de fidelización de usuarios.")
    else:
        st.sidebar.error("No se encontraron suficientes tweets positivos.")


# ============================================================
# 6. FILTRO DE VISUALIZACIÓN
# ============================================================

df_actual = st.session_state.dataset

    # ============================================================
    # MODO DINÁMICO DEL DASHBOARD
    # ============================================================

if "Crisis" in st.session_state.estado:
    modo_dashboard = "crisis"
elif "Campaña" in st.session_state.estado or "Mitigación" in st.session_state.estado:
    modo_dashboard = "campaña"
else:
    modo_dashboard = "monitoreo"

if aerolinea_seleccionada != "Todas":
    df_visualizacion = df_actual[df_actual['Airline'] == aerolinea_seleccionada]
else:
    df_visualizacion = df_actual


# ============================================================
# 7. MÉTRICAS PRINCIPALES
# ============================================================

total_menciones, positivos, negativos, neutros, nsi, porcentaje_negativo = calcular_metricas(df_visualizacion)
riesgo_crisis = calcular_riesgo_crisis(total_menciones, porcentaje_negativo, nsi)
top_topic, top_topic_count = obtener_topico_dominante(df_visualizacion)

color_map = {
    "Positivo": "#2ecc71",
    "Neutro": "#f1c40f",
    "Negativo": "#e74c3c"
}

tab_resumen, tab_engagement, tab_mapa, tab_crisis, tab_feed = st.tabs([
    "📌 Resumen Ejecutivo",
    "📈 Engagement y Tendencias",
    "🌍 Mapa Global",
    "🚨 Crisis / Oportunidades",
    "🧾 Feed"
])


with tab_resumen:

    st.markdown("### Resumen Ejecutivo de Monitoreo")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        label="Tweets Analizados",
        value=total_menciones
    )

    col2.metric(
        label="Negativos",
        value=negativos,
        delta="+150" if "Crisis" in st.session_state.estado else None,
        delta_color="inverse"
    )

    col3.metric(
        label="Positivos",
        value=positivos,
        delta="+150" if "Campaña" in st.session_state.estado else None
    )

    col4.metric(
        label="Net Sentiment",
        value=f"{nsi}%",
        delta="ALERTA" if nsi < -20 else "ESTABLE"
    )

    col5.metric(
        label="Riesgo",
        value=riesgo_crisis
    )

    st.markdown(
        f"**Filtro actual:** `{aerolinea_seleccionada}` | "
        f"**Estado:** `{st.session_state.estado}` | "
        f"**Negatividad:** `{porcentaje_negativo}%`"
    )

    mostrar_semaforo(riesgo_crisis)

    if top_topic:
        st.warning(
            f"Principal foco de crisis detectado: **{top_topic}** "
            f"con **{top_topic_count} menciones negativas**."
        )

    st.markdown("---")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.write("#### Distribución de Sentimientos")

        if total_menciones > 0:
            fig_pie = px.pie(
                df_visualizacion,
                names="Sentiment",
                color="Sentiment",
                color_discrete_map = color_map,
                hole=0.55
            )

            fig_pie.update_layout(
                template="plotly_white",
                height=360,
                margin=dict(l=10, r=10, t=20, b=20),
                legend=dict(orientation="h", y=-0.1)
            )

            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No hay datos para mostrar.")

    with col_b:
        st.write("#### Ranking de Riesgo por Aerolínea")

        df_ranking = generar_ranking_aerolineas(df_actual)

        if not df_ranking.empty:
            st.dataframe(
                df_ranking.head(8),
                use_container_width=True,
                hide_index=True,
                height=360
            )
        else:
            st.info("No hay datos suficientes para generar ranking.")


st.markdown(
    f"**Filtro Actual:** Visibilidad enfocada en `{aerolinea_seleccionada}` | "
    f"**Estado:** `{st.session_state.estado}` | "
    f"**Negatividad:** `{porcentaje_negativo}%`"
)


if top_topic:
    st.warning(
        f"Principal foco de crisis detectado: **{top_topic}** "
        f"con **{top_topic_count} menciones negativas**."
    )

st.markdown("---")

with tab_engagement:

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.write("### Tópicos Críticos o Motivos de Quejas")

        df_topics = df_visualizacion[df_visualizacion['Topic'] != 'General / Other']

        if not df_topics.empty:
            fig_bar = px.bar(
                df_topics,
                y="Topic",
                color="Sentiment",
                color_discrete_map=color_map,
                barmode="stack",
                orientation='h'
            )

            fig_bar.update_layout(
                template="plotly_white",
                height=430,
                margin=dict(l=10, r=10, t=20, b=20),
                yaxis={'categoryorder': 'total ascending'},
                xaxis_title="Cantidad de Tweets"
            )

            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Sin anomalías específicas reportadas.")

    with col_g2:
        st.write("### Evolución Temporal del Sentimiento")

        if not df_visualizacion.empty and "Fecha" in df_visualizacion.columns:
            df_tiempo = df_visualizacion.copy()
            df_tiempo["Fecha"] = pd.to_datetime(df_tiempo["Fecha"], errors="coerce")
            df_tiempo = df_tiempo.dropna(subset=["Fecha"])

            if not df_tiempo.empty:
                df_tiempo = df_tiempo.groupby([
                    pd.Grouper(key="Fecha", freq="H"),
                    "Sentiment"
                ]).size().reset_index(name="Cantidad")

                fig_time = px.line(
                    df_tiempo,
                    x="Fecha",
                    y="Cantidad",
                    color="Sentiment",
                    color_discrete_map=color_map,
                    markers=True
                )

                fig_time.update_layout(
                    template="plotly_white",
                    height=430,
                    margin=dict(l=10, r=10, t=20, b=20),
                    xaxis_title="Tiempo simulado",
                    yaxis_title="Cantidad"
                )

                st.plotly_chart(fig_time, use_container_width=True)
            else:
                st.info("No hay fechas válidas.")
        else:
            st.info("No hay datos suficientes.")

with tab_mapa:

    if modo_dashboard == "crisis":
        st.write("###  Problemas Globales")
        sentimiento_mapa = "Negativo"
        texto_cantidad = "Cantidad de menciones negativas"
    elif modo_dashboard == "campaña":
        st.write("###  Oportunidades Globales")
        sentimiento_mapa = "Positivo"
        texto_cantidad = "Cantidad de menciones positivas"
    else:
        st.write("###  Monitoreo Global")
        sentimiento_mapa = None
        texto_cantidad = "Cantidad de menciones"

    st.caption(
        "Nota metodológica: las ubicaciones son simuladas a partir de hubs principales de aerolíneas "
        "y puntos internacionales agregados para representar monitoreo global. "
        "El dataset original no incluye geolocalización real de los tweets."
    )

    if sentimiento_mapa:
        df_mapa = df_visualizacion[
            (df_visualizacion["Sentiment"] == sentimiento_mapa) &
            (df_visualizacion["Latitud"].notna()) &
            (df_visualizacion["Longitud"].notna())
        ].copy()
    else:
        df_mapa = df_visualizacion[
            (df_visualizacion["Latitud"].notna()) &
            (df_visualizacion["Longitud"].notna())
        ].copy()

    if not df_mapa.empty:

        df_mapa_agg = df_mapa.groupby(
            ["Airline", "Ciudad", "Aeropuerto", "Region", "Latitud", "Longitud", "Sentiment"]
        ).size().reset_index(name="Cantidad")

        color_mapa = "Region" if sentimiento_mapa else "Sentiment"

        fig_mapa = px.scatter_mapbox(
            df_mapa_agg,
            lat="Latitud",
            lon="Longitud",
            size="Cantidad",
            color=color_mapa,
            hover_name="Airline",
            hover_data={
                "Ciudad": True,
                "Aeropuerto": True,
                "Region": True,
                "Sentiment": True,
                "Cantidad": True,
                "Latitud": False,
                "Longitud": False
            },
            zoom=1,
            height=590,
            size_max=55,
            title=texto_cantidad
        )

        fig_mapa.update_layout(
            mapbox_style="carto-darkmatter",
            template="plotly_dark",
            margin=dict(l=5, r=5, t=40, b=5),
            mapbox=dict(
                center=dict(lat=10, lon=10),
                zoom=1.05
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.01,
                xanchor="left",
                x=0.01
            )
        )

        st.plotly_chart(fig_mapa, use_container_width=True)

    else:
        st.info("No hay menciones con ubicación disponible para mostrar en el mapa.")

with tab_crisis:

    if modo_dashboard == "crisis":
        sentimiento_nube = "Negativo"
        titulo_nube = "Menciones Negativas"
    elif modo_dashboard == "campaña":
        sentimiento_nube = "Positivo"
        titulo_nube = "Menciones Positivas"
    else:
        sentimiento_nube = "Negativo"
        titulo_nube = "Menciones Negativas"

    col_nube, col_ranking = st.columns([1.2, 1])

    with col_nube:
        st.write(f"### {titulo_nube}")

        textos_nube = " ".join(
            df_visualizacion[df_visualizacion["Sentiment"] == sentimiento_nube]["Text"].astype(str)
        )

        stopwords_wordcloud = {
            "the", "and", "for", "you", "your", "are", "was", "were", "with",
            "this", "that", "have", "has", "had", "from", "they", "them",
            "then", "than", "there", "their", "will", "would", "could", "should",
            "can", "cant", "can't", "dont", "don't", "did", "didnt", "didn't",
            "not", "but", "all", "our", "out", "get", "got", "now", "why",
            "how", "too", "just", "what", "when", "where", "about", "because",
            "been", "again", "very", "more", "only", "any", "yes", "amp",
            "to", "of", "in", "on", "at", "it", "is", "be", "or", "if",
            "so", "my", "me", "we", "us", "up", "by", "an", "as",
            "http", "https", "co", "rt",
            "flight", "flights", "airline", "airlines", "plane",
            "united", "unitedairlines", "usairways", "americanair",
            "southwestair", "jetblue", "virginamerica", "delta", "southwest",
            "im", "i'm", "ive", "i've"
        }

        if textos_nube.strip():
            wordcloud = WordCloud(
                width=900,
                height=360,
                background_color="#111111",
                colormap="Greens" if sentimiento_nube == "Positivo" else "Reds",
                stopwords=stopwords_wordcloud,
                max_words=70,
                min_word_length=4,
                collocations=False
            ).generate(textos_nube)

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")

            st.pyplot(fig)
        else:
            st.info("No hay suficientes menciones para generar la nube.")

    with col_ranking:
        st.write(f"### Términos Frecuentes - {sentimiento_nube}")

        df_palabras = generar_terminos_frecuentes(df_visualizacion, sentimiento_nube)

        if not df_palabras.empty:
            fig_words = px.bar(
                df_palabras,
                x="Frecuencia",
                y="Palabra",
                orientation="h"
            )

            fig_words.update_layout(
                template="plotly_white",
                height=360,
                margin=dict(l=10, r=10, t=20, b=20),
                yaxis={'categoryorder': 'total ascending'},
                xaxis_title="Frecuencia",
                yaxis_title=""
            )

            st.plotly_chart(fig_words, use_container_width=True)
        else:
            st.info("No hay términos suficientes.")

    st.markdown("---")

    col_rec, col_tablas = st.columns([1, 1.3])

    with col_rec:
        mostrar_recomendacion_dinamica(
            modo_dashboard,
            riesgo_crisis,
            top_topic,
            positivos,
            negativos,
            total_menciones
        )

    with col_tablas:
        if modo_dashboard == "campaña":
            st.info("Modo campaña activo: se priorizan oportunidades positivas.")
        elif modo_dashboard == "crisis":
            st.warning("Modo crisis activo: se priorizan reclamos negativos.")

        st.write("#### Casos Prioritarios")

        df_prioritarios = df_visualizacion[
            (df_visualizacion["Sentiment"] == "Negativo") &
            (df_visualizacion["Topic"] != "General / Other")
        ][["Airline", "User", "Text", "Topic", "Fecha"]].copy()

        df_prioritarios = df_prioritarios.sort_values("Fecha", ascending=False).head(5)

        if not df_prioritarios.empty:
            st.dataframe(
                df_prioritarios,
                use_container_width=True,
                hide_index=True,
                height=210
            )
        else:
            st.success("No hay casos críticos pendientes.")

        st.write("#### Oportunidades de Fidelización")

        df_oportunidades = df_visualizacion[
            df_visualizacion["Sentiment"] == "Positivo"
        ][["Airline", "User", "Text", "Sentiment", "Fecha"]].copy()

        df_oportunidades = df_oportunidades.sort_values("Fecha", ascending=False).head(5)

        if not df_oportunidades.empty:
            st.dataframe(
                df_oportunidades,
                use_container_width=True,
                hide_index=True,
                height=210
            )
        else:
            st.info("No se detectaron oportunidades positivas.")

with tab_feed:

    st.write("### Feed de Ingesta Directa - Clasificación en Tiempo Real")
    st.markdown("Output del pipeline de PLN con clasificación automática de sentimiento y tópico.")

    if not df_visualizacion.empty:
        df_mostrar = df_visualizacion[
            ['Airline', 'User', 'Text', 'Sentiment', 'Topic', 'Fecha']
        ].copy()

        df_mostrar = df_mostrar.sort_values("Fecha", ascending=False)

        styled_df = df_mostrar.style.map(
            color_sentiment_row,
            subset=['Sentiment']
        )

        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            height=620
        )
    else:
        st.info("No hay tweets para mostrar con el filtro actual.")
