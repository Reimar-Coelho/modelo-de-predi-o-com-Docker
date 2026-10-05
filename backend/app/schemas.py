from datetime import date

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    treinado_em: str | None = None
    dados_ate: date | None = None  # último fechamento usado no treino


class PredictRequest(BaseModel):
    janela: int = Field(..., ge=1, le=365, description="dias úteis a prever a partir da última data do treino")


class Ponto(BaseModel):
    data: date
    dias_uteis: int  # h: distância em dias úteis da data base
    previsto: float  # último fechamento (passeio aleatório)
    limite_inferior: float
    limite_superior: float


class PredictResponse(BaseModel):
    moeda: str
    data_base: date  # último dia com fechamento conhecido
    ultimo_fechamento: float
    nivel_intervalo: float  # 0.8 = intervalo de 80%
    previsoes: list[Ponto]
