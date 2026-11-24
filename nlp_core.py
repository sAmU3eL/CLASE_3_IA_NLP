"""Núcleo de la aplicación educativa de NLP (Encuentro 3).

La tokenización de spaCy se muestra solo con fines explicativos.
El clasificador utiliza su propio tokenizador WordPiece y no recibe
los lemas ni el texto sin stop words.
"""

from __future__ import annotations

import math
import re


_ETIQUETA_ESTRELLAS = re.compile(r"([1-5]) stars?", re.IGNORECASE)


def procesar_puntuaciones(predicciones: list) -> list[dict]:
    """Ordena las cinco salidas del clasificador por número de estrellas.

    `pipeline(..., top_k=None)` normalmente devuelve una lista de dicts
    para una única frase. También se tolera el formato anidado antiguo.
    No se convierte el score en una medida de exactitud real del modelo.
    """
    if (
        isinstance(predicciones, list)
        and len(predicciones) == 1
        and isinstance(predicciones[0], list)
    ):
        predicciones = predicciones[0]

    if not isinstance(predicciones, list):
        raise ValueError("La respuesta del modelo no tiene el formato esperado.")

    valores = {}
    for pred in predicciones:
        if not isinstance(pred, dict):
            raise ValueError("El modelo devolvió una predicción inválida.")
        etiqueta = str(pred.get("label", "")).strip()
        coincidencia = _ETIQUETA_ESTRELLAS.fullmatch(etiqueta)
        if coincidencia is None:
            raise ValueError(f"Etiqueta del modelo no reconocida: {etiqueta!r}")
        numero = int(coincidencia.group(1))
        puntaje = float(pred["score"])
        if numero in valores or not math.isfinite(puntaje) or not 0 <= puntaje <= 1:
            raise ValueError("El modelo devolvió etiquetas repetidas o puntuaciones inválidas.")
        valores[numero] = puntaje

    if set(valores) != set(range(1, 6)):
        raise ValueError("Se esperaban puntuaciones para las cinco estrellas (1 a 5).")

    return [
        {
            "estrellas": n,
            "etiqueta": f"{n} star" if n == 1 else f"{n} stars",
            "puntuacion": valores[n],
        }
        for n in range(1, 6)
    ]


def analizar_frase(texto: str, nlp, clasificador, max_length: int = 128) -> dict:
    """Analiza una sola frase y devuelve tokens y las cinco puntuaciones.

    El mismo `max_length` y truncamiento se aplican a los tokens
    visualizados y al texto enviado a la inferencia.
    """
    texto = texto.strip()
    if not texto:
        raise ValueError("Debes ingresar una frase no vacía.")
    if max_length < 3:
        raise ValueError("max_length debe ser al menos 3.")

    doc = nlp(texto)
    tokens_spacy = [
        {
            "Token": token.text,
            "Lema": token.lemma_,
            "Stop word": bool(token.is_stop),
            "Puntuación": bool(token.is_punct),
        }
        for token in doc
    ]

    # Tokens reales del clasificador, que NO son los de BETO del notebook.
    tokenizer = clasificador.tokenizer
    ids_originales = tokenizer(texto, truncation=False, add_special_tokens=True)["input_ids"]
    ids_usados = tokenizer(
        texto, truncation=True, max_length=max_length, add_special_tokens=True
    )["input_ids"]
    tokens_modelo = tokenizer.convert_ids_to_tokens(ids_usados)
    tabla_tokens = [
        {"Posición": i, "Token del modelo": token, "ID": int(token_id)}
        for i, (token, token_id) in enumerate(zip(tokens_modelo, ids_usados))
    ]

    # top_k=None solicita todas las etiquetas en lugar de solo la ganadora.
    predicciones = clasificador(
        texto, truncation=True, max_length=max_length, top_k=None
    )
    distribucion = procesar_puntuaciones(predicciones)
    ganadora = max(distribucion, key=lambda item: item["puntuacion"])

    return {
        "texto": texto,
        "tokens_spacy": tokens_spacy,
        "tokens_modelo": tabla_tokens,
        "total_spacy": len(tokens_spacy),
        "total_modelo": len(ids_usados),
        "total_original_modelo": len(ids_originales),
        "truncado": len(ids_originales) > max_length,
        "etiqueta": ganadora["etiqueta"],
        "puntuacion": ganadora["puntuacion"],
        "distribucion": distribucion,
    }
