# ============================================================
# CABECERA
# ============================================================
# Alumno: Nombre Apellido
# URL Streamlit Cloud: https://...streamlit.app
# URL GitHub: https://github.com/...

# ============================================================
# IMPORTS
# ============================================================
# Streamlit: framework para crear la interfaz web
# pandas: manipulación de datos tabulares
# plotly: generación de gráficos interactivos
# openai: cliente para comunicarse con la API de OpenAI
# json: para parsear la respuesta del LLM (que llega como texto JSON)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
# Modelo de OpenAI. No lo cambies.
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
# El system prompt es el conjunto de instrucciones que recibe el LLM
# ANTES de la pregunta del usuario. Define cómo se comporta el modelo:
# qué sabe, qué formato debe usar, y qué hacer con preguntas inesperadas.
#
# Puedes usar estos placeholders entre llaves — se rellenan automáticamente
# con información real del dataset cuando la app arranca:
#   {fecha_min}             → primera fecha del dataset
#   {fecha_max}             → última fecha del dataset
#   {plataformas}           → lista de plataformas (Android, iOS, etc.)
#   {reason_start_values}   → valores posibles de reason_start
#   {reason_end_values}     → valores posibles de reason_end
#
# IMPORTANTE: como el prompt usa llaves para los placeholders,
# si necesitas escribir llaves literales en el texto (por ejemplo para
# mostrar un JSON de ejemplo), usa doble llave: {{ y }}
#
SYSTEM_PROMPT = """
Eres un asistente analítico que responde preguntas sobre los hábitos de escucha de Spotify de un usuario.

Tienes acceso a un DataFrame de pandas llamado `df` con estos datos:
- Período: desde {fecha_min} hasta {fecha_max}
- Plataformas disponibles: {plataformas}
- reason_start posibles: {reason_start_values}
- reason_end posibles: {reason_end_values}

Columnas disponibles en `df`:
- ts: datetime con timezone UTC
- ms_played: milisegundos reproducidos
- minutos: minutos reproducidos (ms_played / 60000)
- artista: nombre del artista
- cancion: nombre de la canción
- album: nombre del álbum
- platform: plataforma (Android, iOS, Windows, web_player)
- shuffle: booleano, si estaba en modo aleatorio
- skipped: booleano o None, si se saltó la canción
- reason_start: motivo de inicio
- reason_end: motivo de fin
- hora: hora del día (0-23)
- mes: número de mes (1-12)
- mes_nombre: nombre del mes
- dia_semana: día de la semana (0=lunes, 6=domingo)
- es_finde: booleano, True si es sábado o domingo
- trimestre: trimestre del año (1-4)
- primera_escucha: mes (1-12) en que se escuchó por primera vez cada canción

INSTRUCCIONES:
1. Responde SIEMPRE con un JSON válido y nada más. Sin texto antes ni después.
2. Si la pregunta es sobre los datos de escucha, usa este formato:
{{"tipo": "grafico", "codigo": "...", "interpretacion": "..."}}
3. Si la pregunta no tiene nada que ver con música o los datos, usa:
{{"tipo": "fuera_de_alcance", "codigo": "", "interpretacion": "Lo siento, solo puedo responder preguntas sobre tus hábitos de escucha de Spotify."}}

REGLAS PARA EL CÓDIGO:
- Usa siempre Plotly (px o go), nunca matplotlib
- El gráfico final debe guardarse en una variable llamada `fig`
- Usa los nombres de columna exactos que se indican arriba
- El código debe ser ejecutable tal cual, sin imports (ya están disponibles: df, pd, px, go)
- Elige el tipo de gráfico adecuado: barras para rankings, líneas para evolución temporal, pie para proporciones
- Para evolución temporal, agrupa siempre por la columna `mes` (número) y no por `mes_nombre`, para mantener el orden correcto
- Para calcular canciones nuevas por mes: primeras_veces = df.groupby('cancion')['ts'].min().reset_index(); primeras_veces['mes'] = primeras_veces['ts'].dt.tz_convert(None).dt.month; nuevas_por_mes = primeras_veces.groupby('mes').size().reset_index(name='nuevas_canciones')
- Para canciones nuevas por mes usa: nuevas_por_mes = df.groupby('primera_escucha')['cancion'].nunique().reset_index(name='nuevas_canciones'); fig = px.bar(nuevas_por_mes, x='primera_escucha', y='nuevas_canciones')

REGLAS PARA LA INTERPRETACIÓN:
- Máximo 2 frases explicando qué muestra el gráfico
- En español
"""


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
# Esta función se ejecuta UNA SOLA VEZ gracias a @st.cache_data.
# Lee el fichero JSON y prepara el DataFrame para que el código
# que genere el LLM sea lo más simple posible.
#
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")
    
   
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    df["hora"] = df["ts"].dt.hour
    df["mes"] = df["ts"].dt.tz_convert(None).dt.month
    df["mes_nombre"] = df["ts"].dt.strftime("%B")
    df["dia_semana"] = df["ts"].dt.dayofweek  
    df["es_finde"] = df["dia_semana"].isin([5, 6])
    df["trimestre"] = df["ts"].dt.quarter
    df["minutos"] = df["ms_played"] / 60000
    df["artista"] = df["master_metadata_album_artist_name"]
    df["cancion"] = df["master_metadata_track_name"]
    df["album"] = df["master_metadata_album_album_name"]
    df["primera_escucha"] = df.groupby("cancion")["mes"].transform("min")

    
    return df


def build_prompt(df):
    """
    Inyecta información dinámica del dataset en el system prompt.
    Los valores que calcules aquí reemplazan a los placeholders
    {fecha_min}, {fecha_max}, etc. dentro de SYSTEM_PROMPT.

    Si añades columnas nuevas en load_data() y quieres que el LLM
    conozca sus valores posibles, añade aquí el cálculo y un nuevo
    placeholder en SYSTEM_PROMPT.
    """
    fecha_min = df["ts"].min()
    fecha_max = df["ts"].max()
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# Esta función envía DOS mensajes a la API de OpenAI:
# 1. El system prompt (instrucciones generales para el LLM)
# 2. La pregunta del usuario
#
# El LLM devuelve texto (que debería ser un JSON válido).
# temperature=0.2 hace que las respuestas sean más predecibles.
#
# No modifiques esta función.
#
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# El LLM devuelve un string que debería ser un JSON con esta forma:
#
#   {"tipo": "grafico",          "codigo": "...", "interpretacion": "..."}
#   {"tipo": "fuera_de_alcance", "codigo": "",    "interpretacion": "..."}
#
# Esta función convierte ese string en un diccionario de Python.
# Si el LLM envuelve el JSON en backticks de markdown (```json...```),
# los limpia antes de parsear.
#
# No modifiques esta función.
#
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# El LLM genera código Python como texto. Esta función lo ejecuta
# usando exec() y busca la variable `fig` que el código debe crear.
# `fig` debe ser una figura de Plotly (px o go).
#
# El código generado tiene acceso a: df, pd, px, go.
#
# No modifiques esta función.
#
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
# Toda la interfaz de usuario. No modifiques esta sección.
#

# Configuración de la página
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
# Lee la contraseña de secrets.toml. Si no coincide, no muestra la app.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

# Cargar datos y construir el prompt con información del dataset
df = load_data()
system_prompt = build_prompt(df)

# Caja de texto para la pregunta del usuario
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    # Mostrar la pregunta en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                # 1. Enviar pregunta al LLM
                raw = get_response(prompt, system_prompt)

                # 2. Parsear la respuesta JSON
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    # Pregunta fuera de alcance: mostrar solo texto
                    st.write(parsed["interpretacion"])
                else:
                    # Pregunta válida: ejecutar código y mostrar gráfico
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# Responde a estas tres preguntas con tus palabras. Sé concreto
# y haz referencia a tu solución, no a generalidades.
# No superes las 30 líneas en total entre las tres respuestas.
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    ¿Cómo funciona la arquitectura de tu aplicación? ¿Qué recibe
#    el LLM? ¿Qué devuelve? ¿Dónde se ejecuta el código generado?
#    ¿Por qué el LLM no recibe los datos directamente?
#
#    [Tu respuesta aquí]
#
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    ¿Qué información le das al LLM y por qué? Pon un ejemplo
#    concreto de una pregunta que funciona gracias a algo específico
#    de tu prompt, y otro de una que falla o fallaría si quitases
#    una instrucción.
#
#    [Tu respuesta aquí]
#
#
# 3. EL FLUJO COMPLETO
#    Describe paso a paso qué ocurre desde que el usuario escribe
#    una pregunta hasta que ve el gráfico en pantalla.
#
#    [Tu respuesta aquí]