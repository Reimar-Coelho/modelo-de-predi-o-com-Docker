# Passos 8-9 do diagrama: carrega o modelo gerado pelo contêiner de treino.
# Passo 13: predict().

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# no compose, aponta para o volume compartilhado com o contêiner de treino
PASTA_MODELO = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parents[2] / "models"))

modelo = None
metadata: dict = {}


def carregar():
    global modelo, metadata
    modelo = joblib.load(PASTA_MODELO / "modelo.joblib")
    metadata = json.loads((PASTA_MODELO / "metadata.json").read_text())


def nivel_intervalo() -> float:
    # quantis (0.1, 0.9) -> intervalo de 80%
    inferior, superior = sorted(modelo["quantis"])
    return round(superior - inferior, 2)


def prever(janela: int) -> list[dict]:
    # janela = dias úteis a prever a partir da última data do treino.
    # previsão central = último fechamento; intervalo = quantis do retorno de 1 dia
    datas = pd.bdate_range(pd.Timestamp(modelo["ultima_data"]) + pd.offsets.BDay(1), periods=janela)
    ultimo = modelo["ultimo_fechamento"]
    inferior, superior = (
        modelo["quantis"][q].predict(modelo["features_ultimo_dia"])[0] for q in sorted(modelo["quantis"])
    )
    raiz_h = np.sqrt(np.arange(1, janela + 1))
    return [
        {
            "data": data.date(),
            "dias_uteis": h,
            "previsto": ultimo,
            "limite_inferior": float(ultimo * np.exp(inferior * r)),
            "limite_superior": float(ultimo * np.exp(superior * r)),
        }
        for h, (data, r) in enumerate(zip(datas, raiz_h), start=1)
    ]
