# EURUSD=X

import time
from pathlib import Path

import yfinance as yf

caminho = Path(__file__).resolve().parent.parent / "data" / "dados_eurusd.csv"
caminho.parent.mkdir(parents=True, exist_ok=True)

# o Yahoo às vezes devolve vazio ("possibly delisted") e funciona na tentativa seguinte
for tentativa in range(1, 6):
    dados = yf.download("EURUSD=X", start="2010-01-01", end="2026-10-01")
    if not dados.empty:
        break
    print(f"tentativa {tentativa}: download vazio, tentando de novo")
    time.sleep(2 * tentativa)
else:
    # não sobrescreve o CSV existente com um arquivo vazio
    raise SystemExit("o yfinance não devolveu dados; o CSV atual foi mantido")

dados.to_csv(caminho)
print(f"{len(dados)} linhas salvas em {caminho}")
