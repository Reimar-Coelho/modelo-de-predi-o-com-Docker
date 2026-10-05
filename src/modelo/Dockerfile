FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# mesmo caminho do repo (src/modelo/), para o RAIZ do script continuar válido
COPY treinar-modelo.py src/modelo/

# UID 1000 = usuário do host: os arquivos gravados em ./models não ficam como root
RUN useradd --create-home --uid 1000 app
USER app

# dados e modelo chegam por volumes montados pelo compose
ENV DATA_PATH=/data/dados_eurusd.csv \
    MODEL_DIR=/models

CMD ["python", "src/modelo/treinar-modelo.py"]
