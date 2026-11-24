# NLP Lab · Tokens y predicción

Aplicación educativa basada en el notebook **Encuentro 3: Representación de texto y selección de modelos**.

El usuario introduce **una sola frase** y obtiene:

1. La etiqueta de sentimiento (1 a 5 estrellas) y su puntuación.
2. **Un gráfico con las puntuaciones de las cinco estrellas** y una tabla numérica.
3. Los **tokens reales e IDs** del tokenizador del clasificador.
4. Una comparación opcional de la tokenización de spaCy.
5. Una explicación integrada de **tokenización vs. predicción**, paso por paso.

> El modelo `nlptown/bert-base-multilingual-uncased-sentiment` fue ajustado con reseñas de productos. Sus puntuaciones son las salidas de la clasificación entre estrellas; **no son precisión o certeza calibrada**. El modelo no clasifica motivos de solicitudes de estudiantes.

## Relación con el notebook

| Notebook | Archivo de la aplicación | Función |
|---|---|---|
| Celdas 8–9 · spaCy | `nlp_core.py` | Tokens, lemas, stop words y puntuación |
| Celdas 20–22 · BETO | Excluido | Experimento aparte, otro tokenizador |
| Celda 28 · Hugging Face | `app.py`, `nlp_core.py` | Carga y predicción con el modelo de sentimiento |
| Celda 31 · `cadena()` | `nlp_core.py` | Integración de análisis y predicción |

Para obtener las cinco puntuaciones, ahora se utiliza `top_k=None` en la llamada al clasificador. Solo se muestra la predicción de mayor puntuación como etiqueta ganadora.

## Estructura del repositorio

```text
nlp-lab-streamlit/
├── app.py
├── nlp_core.py
├── requirements.txt
├── README.md
├── tests/
│   └── test_nlp_core.py
├── .gitignore
└── .streamlit/
    └── config.toml
```

## Publicar en Streamlit Community Cloud

1. Sube el contenido de la carpeta (no el ZIP) a la **raíz** del repositorio GitHub.
2. Abre <https://share.streamlit.io/> y vincula el repositorio.
3. Selecciona la rama `main`, archivo principal `app.py` y Python **3.12** en *Advanced settings*.
4. Publica la aplicación. `requirements.txt` define la versión de Streamlit y el resto de bibliotecas.
5. La primera consulta descarga el modelo de Hugging Face (requiere acceso a Internet); pueden darse límites de memoria o problemas de conexión en la instancia.

## Verificar la lógica (sin descargar modelos)

```bash
python -m unittest discover -s tests -v
```

Las pruebas utilizan sustitutos del modelo real para comprobar los resultados, las cinco puntuaciones, el truncamiento y la conservación de negaciones. **No sustituyen una prueba completa en Streamlit Community Cloud.**
