# ============================================================
# CABECERA
# ============================================================
# Alumno: Pablo Fernandez Segura
# URL Streamlit Cloud: https://spotify-analytics-stats-pfs.streamlit.app
# URL GitHub: https://github.com/pfernandez-sketch/spotify-analytics

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
- horas: horas reproducidas (ms_played / 3600000)
- artista: nombre del artista
- cancion: nombre de la canción
- album: nombre del álbum
- platform: plataforma (Android, iOS, Windows, web_player)
- shuffle: booleano, si estaba en modo aleatorio
- skipped: booleano o None, si se saltó la canción
- skipped_bool: booleano limpio, True si se saltó la canción y False si no
- reason_start: motivo de inicio
- reason_end: motivo de fin
- hora: hora del día (0-23)
- mes: número de mes (1-12)
- mes_nombre: nombre del mes
- dia_semana: día de la semana (0=lunes, 6=domingo)
- es_finde: booleano, True si es sábado o domingo
- trimestre: trimestre del año (1-4)
- semestre: primer_semestre o segundo_semestre
- estacion: invierno, primavera, verano u otoño
- primera_escucha: mes (1-12) en que se escuchó por primera vez cada canción

INSTRUCCIONES:
1. Responde SIEMPRE con un JSON válido y nada más. No escribas texto fuera del JSON.
2. Si la pregunta es sobre los datos de escucha, usa exactamente este formato:
{{"tipo": "grafico", "codigo": "...", "interpretacion": "..."}}
3. Si la pregunta no está relacionada con los hábitos de escucha de Spotify, usa exactamente este formato:
{{"tipo": "fuera_de_alcance", "codigo": "", "interpretacion": "Lo siento, solo puedo responder preguntas sobre tus hábitos de escucha de Spotify."}}
4. No inventes columnas, datos, filtros ni cálculos que no existan en el DataFrame.
5. No pidas aclaraciones. Si la pregunta es ambigua pero razonable, elige la interpretación más natural para analizar hábitos de escucha.

REGLAS PARA EL CÓDIGO:
- Usa siempre Plotly (px o go), nunca matplotlib.
- El gráfico final debe guardarse en una variable llamada `fig`.
- Usa únicamente las variables ya disponibles: `df`, `pd`, `px`, `go`.
- NUNCA escribas líneas import ni from ... import. Las variables df, pd, px y go ya están disponibles directamente sin importar nada. Si escribes cualquier import el código fallará con error.
- El código debe ser ejecutable tal cual, sin comentarios y sin texto explicativo.
- Usa exactamente los nombres de columna indicados arriba.
- Añade siempre un título y etiquetas claras en español.
- Ordena los rankings de mayor a menor.
- Usa gráficos de barras para rankings y comparaciones, líneas para evolución temporal. Usa pie chart cuando haya 2 o 3 categorías y la pregunta sea sobre proporción o distribución (shuffle vs orden, semestres, saltadas vs no saltadas).
- Si el usuario pregunta por un único elemento ("cuál es", "qué canción", "qué artista", "qué plataforma"), devuelve solo el top 1.
- Usa top 5 o top 10 solo cuando el usuario lo pida explícitamente o cuando la pregunta esté en plural.
- Si el usuario habla de tiempo escuchado, usa `horas` o `minutos`.
- Si el usuario habla de número de reproducciones, usa recuento de filas.
- Si la pregunta es "más escuchado" y no especifica si quiere tiempo o reproducciones, interpreta "más escuchado" como tiempo total reproducido.
- Para preguntas sobre canciones saltadas, usa siempre `skipped_bool`, nunca `skipped`.
- Para preguntas sobre shuffle vs orden, puedes comparar por reproducciones salvo que el usuario pida explícitamente tiempo.
- Para evolución temporal, agrupa por `mes` y no por `mes_nombre`, para mantener el orden correcto.
- Cuando el eje X represente meses (1-12), convierte esa columna a string antes de graficar para que aparezca como categoría.
- Para gráficos por hora del día, asegúrate siempre de que aparezcan las 24 horas (0-23). Crea un DataFrame base con todas las horas usando range(24), haz un merge con los datos reales y rellena con 0 las horas sin datos. Así ninguna hora queda fuera del gráfico.
- Para comparaciones entre periodos (semestre, estación, entre semana vs fin de semana), usa gráficos comparativos claros, preferiblemente barras agrupadas.
- Si el usuario pide comparar top artistas o top canciones entre dos periodos, calcula el top de cada periodo, une los elementos relevantes en un único DataFrame comparativo y representa ambas series de forma clara.
- Para gráficos por trimestre, agrupa por `trimestre` directamente sin reindexar y convierte a string antes de graficar: df_trim = df.groupby('trimestre')['horas'].sum().reset_index(); df_trim['trimestre'] = df_trim['trimestre'].astype(str)
- Para gráficos de evolución por mes, usa `mes_nombre` en el eje X en lugar de `mes`, pero ordena el DataFrame por `mes` antes de graficar para mantener el orden correcto: df_mes = df.groupby(['mes','mes_nombre'])['minutos'].sum().reset_index().sort_values('mes')
- Para preguntas sobre shuffle vs orden y sobre comparaciones entre dos periodos como semestres, usa siempre pie chart.
- Para preguntas sobre horas del día, usa siempre gráfico de barras (px.bar), nunca líneas.
- Para canciones nuevas por mes, crea un DataFrame con los 12 meses usando el mapa de mes_nombre, haz merge con los datos reales, rellena con 0 los meses sin datos y muestra todos los meses de Enero a Diciembre en orden.

REGLAS ESPECÍFICAS ÚTILES:
- Para "¿en qué mes descubrí más canciones nuevas?", calcula canciones únicas por `primera_escucha`, ordena por el mes y grafica el resultado por mes.
- Para preguntas sobre entre semana, usa `es_finde == False`.
- Para preguntas sobre fines de semana, usa `es_finde == True`.
- Para preguntas sobre estaciones, usa `estacion`.
- Para preguntas sobre semestres, usa `semestre`.

REGLAS PARA LA INTERPRETACIÓN:
- Escribe la interpretación en español.
- Máximo 2 frases.
- No describas solo el gráfico: menciona también el hallazgo principal cuando sea evidente.
- Si la respuesta permite identificar claramente un ganador, pico o categoría dominante, menciónalo explícitamente.
- Si hay una categoría dominante o un periodo claramente superior, dilo de forma directa.
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
    df = df[df["ms_played"] > 0].copy()

    df["hora"] = df["ts"].dt.hour
    df["mes"] = df["ts"].dt.month
    df["dia_semana"] = df["ts"].dt.dayofweek
    df["es_finde"] = df["dia_semana"].isin([5, 6])
    df["trimestre"] = df["ts"].dt.quarter
    df["minutos"] = df["ms_played"] / 60000
    df["horas"] = df["ms_played"] / 3600000

    df["artista"] = df["master_metadata_album_artist_name"]
    df["cancion"] = df["master_metadata_track_name"]
    df["album"] = df["master_metadata_album_album_name"]

    mapa_meses = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
        5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
        9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
    }
    df["mes_nombre"] = df["mes"].map(mapa_meses)
    df = df[df["artista"].notna() & df["cancion"].notna()].copy()
    df["skipped_bool"] = df["skipped"].fillna(False).astype(bool)
    df["semestre"] = df["mes"].apply(lambda x: "primer_semestre" if x <= 6 else "segundo_semestre")

    def asignar_estacion(mes):
        if mes in [12, 1, 2]:
            return "invierno"
        elif mes in [3, 4, 5]:
            return "primavera"
        elif mes in [6, 7, 8]:
            return "verano"
        else:
            return "otoño"

    df["estacion"] = df["mes"].apply(asignar_estacion)

    primera_ts = df.groupby("spotify_track_uri")["ts"].transform("min")
    df["primera_escucha"] = primera_ts.dt.month

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
#    En mi aplicación el LLM no recibe el dataset real ni las filas del historial, sino una descripción estructurada del DataFrame: columnas disponibles, 
#    rango temporal, plataformas y valores posibles de algunas variables. A partir de esa información y de la pregunta del usuario, el modelo devuelve un JSON 
#    con tres campos: `tipo`, `codigo` e `interpretacion`. El código generado se ejecuta en local con `exec()` dentro de `execute_chart()`, usando el DataFrame ya cargado en memoria. 
#    Esta arquitectura evita exponer datos reales a la API y hace que el análisis se haga sobre el fichero local, no dentro del modelo. El LLM actúa como generador de código, no como 
#    motor que “ve” directamente los datos.
#

# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    ¿Qué información le das al LLM y por qué? Pon un ejemplo
#    concreto de una pregunta que funciona gracias a algo específico
#    de tu prompt, y otro de una que falla o fallaría si quitases
#    una instrucción.

#    El system prompt es la pieza que más condiciona la calidad de la app, porque le dice al modelo qué columnas existen, qué significan y cómo debe responder. En mi caso le indico columnas 
#    derivadas como `horas`, `skipped_bool`, `semestre`, `estacion` y `primera_escucha`, además de reglas para usar siempre JSON válido, guardar el gráfico en `fig`, no escribir imports y 
#    elegir gráficos adecuados. Por ejemplo, la pregunta “¿Qué porcentaje de canciones salto?” funciona bien gracias a que el prompt obliga a usar `skipped_bool` en lugar de `skipped`, que 
#    tiene nulos. También mejoró mucho “¿Qué canción he escuchado más veces?” cuando añadí la instrucción de devolver top 1 si la pregunta está en singular. Si quitase la instrucción de responder 
#    solo con JSON o la de no inventar columnas, la app fallaría al parsear la respuesta o al ejecutar código incorrecto.

# 3. EL FLUJO COMPLETO
#    Describe paso a paso qué ocurre desde que el usuario escribe
#    una pregunta hasta que ve el gráfico en pantalla.
#
#    El flujo empieza cuando el usuario escribe una pregunta en `st.chat_input`. La app carga el DataFrame ya preparado con `load_data()` y construye el prompt final con `build_prompt()`, 
#    insertando fechas, plataformas y valores reales del dataset. Después envía a la API dos mensajes: el system prompt y la pregunta del usuario. La respuesta del modelo llega como texto y 
#    `parse_response()` la convierte en un diccionario Python. Si el tipo es `fuera_de_alcance`, la app muestra solo el mensaje controlado. Si el tipo es `grafico`, `execute_chart()` ejecuta el
#     código generado sobre `df`, recupera la figura `fig` y Streamlit la muestra junto con la interpretación y el código utilizado. Así, cada pregunta se resuelve de forma independiente, sin memoria conversacional
#     y sin sacar los datos fuera del entorno local.