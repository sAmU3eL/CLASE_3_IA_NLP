"""Interfaz de NLP Lab: escribir frase → tokens → cinco puntuaciones.

Para desplegar: subir este repositorio a GitHub y seleccionar `app.py`
como archivo principal en Streamlit Community Cloud.
"""

import pandas as pd
import streamlit as st

from nlp_core import analizar_frase


MODELO_SENTIMIENTO = "nlptown/bert-base-multilingual-uncased-sentiment"
MAX_TOKENS = 128

st.set_page_config(
    page_title="NLP Lab · Tokens y predicción",
    page_icon="🔤",
    layout="centered",
)


@st.cache_resource(show_spinner="Cargando spaCy en español...")
def cargar_spacy():
    import spacy

    return spacy.load("es_core_news_sm")


@st.cache_resource(show_spinner="Cargando clasificador de sentimientos...")
def cargar_clasificador():
    import torch
    from transformers import pipeline

    # CPU: compatible con las instancias gratuitas de Community Cloud.
    torch.set_num_threads(2)
    return pipeline(
        task="text-classification",
        model=MODELO_SENTIMIENTO,
        device=-1,
    )


st.title("NLP Lab: de una frase a una predicción")
st.write(
    "Explora cómo una frase se convierte en **tokens**, cómo esos tokens "
    "se representan con **identificadores numéricos** y cómo un modelo "
    "Transformer produce **cinco puntuaciones de sentimiento**."
)

with st.form("formulario_frase"):
    frase = st.text_area(
        "Frase para analizar",
        value="No me gustó la clase de hoy",
        max_chars=4000,
        height=110,
        help="El texto se analiza tal como lo escribes: no eliminamos negaciones.",
    )
    analizar = st.form_submit_button("Mostrar tokens y predicción", type="primary")

if analizar:
    if not frase.strip():
        st.warning("Ingresa una frase antes de analizar.")
    else:
        try:
            with st.spinner("Analizando frase..."):
                nlp = cargar_spacy()
                clasificador = cargar_clasificador()
                salida = analizar_frase(frase, nlp, clasificador, MAX_TOKENS)
        except Exception as exc:
            st.error("El análisis no pudo completarse. Revisa los logs del despliegue.")
            with st.expander("Detalle técnico del error"):
                st.exception(exc)
        else:
            st.divider()
            st.subheader("1. Predicción del modelo")
            col1, col2 = st.columns(2)
            col1.metric("Etiqueta predicha", salida["etiqueta"])
            col2.metric("Puntuación de esa etiqueta", f'{salida["puntuacion"]:.1%}')

            st.markdown("#### Puntuación de las cinco etiquetas")
            puntajes = pd.DataFrame(
                [
                    {
                        "Estrellas": f'{fila["estrellas"]} ★',
                        "Puntuación (%)": round(fila["puntuacion"] * 100, 2),
                    }
                    for fila in salida["distribucion"]
                ]
            )
            st.bar_chart(
                puntajes.set_index("Estrellas"),
                y="Puntuación (%)",
                horizontal=True,
            )
            with st.expander("Ver las cinco puntuaciones exactas"):
                st.dataframe(puntajes, hide_index=True, width="stretch")
            st.caption(
                "Estas cinco puntuaciones suman aproximadamente 100 % (salvo redondeo). "
                "El modelo elige la categoría con la mayor puntuación. "
                "No son porcentajes de positividad ni una garantía de acierto."
            )
            st.info(
                "El clasificador fue ajustado con reseñas de productos y devuelve 1–5 estrellas. "
                "Una frase educativa no es necesariamente una reseña; interpreta con cautela "
                "las estrellas y sus puntuaciones."
            )

            st.subheader("2. Tokens realmente utilizados por el clasificador")
            st.write([fila["Token del modelo"] for fila in salida["tokens_modelo"]])
            st.dataframe(
                salida["tokens_modelo"],
                hide_index=True,
                width="stretch",
            )
            st.caption(
                f'Tokens enviados al modelo (incluidos los especiales): {salida["total_modelo"]}. '
                f'Tokens sin truncar: {salida["total_original_modelo"]}.'
            )
            if salida["truncado"]:
                st.warning(
                    f'La frase superó el límite de {MAX_TOKENS} tokens. '
                    "El clasificador recibió la versión truncada del texto."
                )

            with st.expander("3. Comparación con spaCy (notebook)"):
                st.write([fila["Token"] for fila in salida["tokens_spacy"]])
                st.dataframe(
                    salida["tokens_spacy"],
                    hide_index=True,
                    width="stretch",
                )
                st.caption(
                    "spaCy divide palabras y puntuación y calcula lemas. "
                    "Sus tokens no se envían al clasificador de sentimientos."
                )

            st.subheader("4. ¿Qué diferencia hay entre tokenización y predicción?")
            st.markdown(
                "**Tokenización:** convierte una frase en unidades pequeñas "
                "(palabras o subpalabras). Por ejemplo, `gusta` puede dividirse "
                "en `gus` y `##ta`. Cada token tiene un ID dentro del vocabulario. "
                "**La tokenización no determina el sentimiento.**"
            )
            st.markdown(
                "**Predicción:** el modelo convierte los IDs en *embeddings*, "
                "calcula representaciones contextualizadas con capas Transformer "
                "y obtiene una puntuación para cada una de sus cinco etiquetas. "
                "La etiqueta con mayor puntuación es la respuesta mostrada."
            )
            st.code(
                "Frase original\n"
                "    ↓\n"
                "Tokenizador del clasificador → tokens + IDs\n"
                "    ↓\n"
                "Embeddings + Transformer\n"
                "    ↓\n"
                "Cinco puntuaciones → etiqueta con mayor puntuación",
                language="text",
            )
            st.caption(
                "Relación con el notebook: spaCy (celdas 8–9); "
                "clasificación Hugging Face (celda 28); integración (celda 31). "
                "El tokenizador BETO de las celdas 20–22 es un experimento distinto."
            )

st.divider()
st.caption(
    "Basado en Encuentro 3 · Representación de texto y selección de modelos. "
    "No se entrena un modelo nuevo: se utiliza uno preentrenado de Hugging Face."
)
