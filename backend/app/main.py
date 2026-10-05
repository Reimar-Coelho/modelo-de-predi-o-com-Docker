from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app import modelo
from app.schemas import HealthResponse, Ponto, PredictRequest, PredictResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # o compose só sobe a api depois que o treino termina (depends_on)
    modelo.carregar()
    yield


app = FastAPI(title="API de predição EURUSD", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if modelo.modelo else "sem_modelo",
        treinado_em=modelo.metadata.get("treinado_em"),
        dados_ate=modelo.modelo["ultima_data"] if modelo.modelo else None,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not modelo.modelo:
        raise HTTPException(status_code=503, detail="modelo não carregado")

    return PredictResponse(
        moeda=modelo.metadata["moeda"],
        data_base=modelo.modelo["ultima_data"],
        ultimo_fechamento=modelo.modelo["ultimo_fechamento"],
        nivel_intervalo=modelo.nivel_intervalo(),
        previsoes=[Ponto(**p) for p in modelo.prever(req.janela)],
    )
