import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import QuantileRegressor

RAIZ = Path(__file__).resolve().parents[2]  # src/modelo/ -> raiz do repo
CAMINHO_DADOS = Path(os.getenv("DATA_PATH", RAIZ / "data" / "dados_eurusd.csv"))
PASTA_MODELO = Path(os.getenv("MODEL_DIR", RAIZ / "models"))

INICIO_VALIDACAO = pd.Timestamp("2024-01-01")
INICIO_TESTE = pd.Timestamp("2025-01-01")
QUANTIS = (0.1, 0.9)  # intervalo de 80%
JANELAS_VOL = (5, 20, 60)
HORIZONTES_AVALIACAO = (1, 5, 21, 63)  # 1 dia, 1 semana, 1 mês, 1 trimestre (dias úteis)


def carregar_dados(caminho):
    # o CSV do yfinance tem 3 linhas de cabeçalho: Price / Ticker / Date
    df = pd.read_csv(caminho, skiprows=[1, 2])
    df = df.rename(columns={"Price": "ds", "Close": "y"})[["ds", "y"]]
    df["ds"] = pd.to_datetime(df["ds"])
    return df.dropna().sort_values("ds").reset_index(drop=True)


def calcular_features(fechamentos):
    # desvio padrão dos retornos log nas últimas 5, 20 e 60 sessões
    retornos = np.log(fechamentos).diff()
    return pd.DataFrame({f"vol{w}": retornos.rolling(w).std() for w in JANELAS_VOL})


def treinar(X, retorno_amanha):
    return {
        q: QuantileRegressor(quantile=q, alpha=0.0, solver="highs").fit(X, retorno_amanha)
        for q in QUANTIS
    }


def intervalo_log(modelos, X, h):
    # quantis do retorno log acumulado em h dias úteis
    return [modelos[q].predict(X) * np.sqrt(h) for q in QUANTIS]


def metricas_intervalo(real, inferior, superior):
    def pinball(q, a):
        d = real - q
        return np.mean(np.maximum(a * d, (a - 1) * d))

    return {
        "cobertura": float(np.mean((real >= inferior) & (real <= superior))),
        "largura_media_pct": float(np.mean(np.exp(superior) - np.exp(inferior)) * 100),
        "pinball_bp": float((pinball(inferior, QUANTIS[0]) + pinball(superior, QUANTIS[1])) / 2 * 1e4),
    }


def avaliar(df, X, fim_treino, inicio, fim):
    """Treina com dados anteriores a fim_treino e avalia as previsões feitas em [inicio, fim)."""
    log_preco = np.log(df["y"])
    tem_features = X.notna().all(axis=1)
    # só entram no treino retornos que terminam antes de fim_treino (sem vazamento)
    retorno_1 = log_preco.shift(-1) - log_preco
    treino = tem_features & (df["ds"].shift(-1) < fim_treino)
    modelos = treinar(X[treino], retorno_1[treino])

    resultado = {"treino_ate": str((fim_treino - pd.Timedelta(days=1)).date()), "inicio": str(inicio.date()), "por_horizonte": {}}
    for h in HORIZONTES_AVALIACAO:
        retorno_h = log_preco.shift(-h) - log_preco
        origens = tem_features & retorno_h.notna() & (df["ds"] >= inicio) & (df["ds"] < fim)
        real = retorno_h[origens].values
        inferior, superior = intervalo_log(modelos, X[origens], h)
        # baseline: quantis fixos dos retornos de h dias vistos no treino
        historico = retorno_h[tem_features & (df["ds"].shift(-h) < fim_treino)]
        fixo = [np.full(len(real), np.quantile(historico, q)) for q in QUANTIS]
        resultado["por_horizonte"][str(h)] = {
            "previsoes": int(origens.sum()),
            "mae_preco": float(np.abs(df["y"].shift(-h)[origens] - df["y"][origens]).mean()),
            "modelo": metricas_intervalo(real, inferior, superior),
            "baseline_intervalo_fixo": metricas_intervalo(real, *fixo),
        }
    return resultado, modelos


def imprimir(nome, resultado):
    print(f"\n{nome} (treino até {resultado['treino_ate']}):")
    print(f"  {'h':>3} {'n':>4} | {'cobertura':>9} {'largura':>8} {'pinball':>8} | baseline fixo: {'cobertura':>9} {'largura':>8} {'pinball':>8}")
    for h, r in resultado["por_horizonte"].items():
        m, b = r["modelo"], r["baseline_intervalo_fixo"]
        print(
            f"  {h:>3} {r['previsoes']:>4} | {m['cobertura']:>9.1%} {m['largura_media_pct']:>7.2f}% {m['pinball_bp']:>8.2f} |"
            f"                {b['cobertura']:>9.1%} {b['largura_media_pct']:>7.2f}% {b['pinball_bp']:>8.2f}"
        )


def salvar_grafico(df, X, modelos_teste, modelos_final, caminho):
    teste = (df["ds"] >= INICIO_TESTE) & X.notna().all(axis=1)
    origens = teste & df["y"].shift(-1).notna()
    inferior, superior = intervalo_log(modelos_teste, X[origens], 1)
    datas_alvo = df["ds"].shift(-1)[origens]
    preco_origem = df["y"][origens].values
    retorno_real = (np.log(df["y"]).shift(-1) - np.log(df["y"]))[origens].values
    fora = (retorno_real < inferior) | (retorno_real > superior)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

    ax1.plot(df["ds"][teste], df["y"][teste], color="black", lw=1, label="fechamento real")
    ax1.fill_between(datas_alvo, preco_origem * np.exp(inferior), preco_origem * np.exp(superior), color="tab:blue", alpha=0.3, label="intervalo 80% previsto na véspera")
    ax1.set_title("Teste: intervalo previsto para o dia seguinte")
    ax1.set_ylabel("USD por EUR")
    ax1.legend()

    ax2.plot(datas_alvo, inferior * 100, color="tab:blue", lw=1)
    ax2.plot(datas_alvo, superior * 100, color="tab:blue", lw=1, label="limites previstos (10% / 90%)")
    ax2.scatter(datas_alvo[~fora], retorno_real[~fora] * 100, s=6, color="gray", label="retorno real dentro")
    ax2.scatter(datas_alvo[fora], retorno_real[fora] * 100, s=8, color="tab:red", label=f"retorno real fora ({fora.mean():.0%})")
    ax2.set_title("Teste: retorno diário e limites previstos (o intervalo acompanha a volatilidade)")
    ax2.set_ylabel("retorno (%)")
    ax2.legend(loc="lower left")

    # o que a API devolve: os próximos 252 dias úteis a partir do último dado
    ultimo, ultima_data = df["y"].iloc[-1], df["ds"].iloc[-1]
    h = np.arange(1, 253)
    datas_futuras = pd.bdate_range(ultima_data + pd.offsets.BDay(1), periods=len(h))
    inf_f, sup_f = (modelos_final[q].predict(X.iloc[[-1]])[0] * np.sqrt(h) for q in QUANTIS)
    recente = df["ds"] >= ultima_data - pd.DateOffset(years=1)
    ax3.plot(df["ds"][recente], df["y"][recente], color="black", lw=1, label="histórico")
    ax3.plot(datas_futuras, np.full(len(h), ultimo), color="tab:blue", label="previsão central")
    ax3.fill_between(datas_futuras, ultimo * np.exp(inf_f), ultimo * np.exp(sup_f), color="tab:blue", alpha=0.3, label="intervalo 80%")
    ax3.set_title(f"Modelo final: previsão para os próximos 252 dias úteis a partir de {ultima_data.date()}")
    ax3.set_ylabel("USD por EUR")
    ax3.legend(loc="upper left")

    fig.suptitle("EURUSD=X — passeio aleatório + regressão quantílica da volatilidade", fontsize=13)
    fig.tight_layout()
    fig.savefig(caminho, dpi=110)
    plt.close(fig)


def main():
    df = carregar_dados(CAMINHO_DADOS)
    X = calcular_features(df["y"])
    print(f"Dados: {len(df)} dias úteis, de {df['ds'].min().date()} a {df['ds'].max().date()}")

    validacao, _ = avaliar(df, X, INICIO_VALIDACAO, INICIO_VALIDACAO, INICIO_TESTE)
    teste, modelos_teste = avaliar(df, X, INICIO_TESTE, INICIO_TESTE, df["ds"].max() + pd.Timedelta(days=1))
    imprimir("Validação 2024", validacao)
    imprimir(f"Teste {INICIO_TESTE.date()} em diante", teste)

    # modelo final: todo o histórico com retorno do dia seguinte conhecido
    log_preco = np.log(df["y"])
    retorno_1 = log_preco.shift(-1) - log_preco
    linhas = X.notna().all(axis=1) & retorno_1.notna()
    modelos_final = treinar(X[linhas], retorno_1[linhas])

    PASTA_MODELO.mkdir(parents=True, exist_ok=True)
    salvar_grafico(df, X, modelos_teste, modelos_final, PASTA_MODELO / "avaliacao_teste.png")

    features_hoje = X.iloc[[-1]].reset_index(drop=True)
    ultimo_fechamento = float(df["y"].iloc[-1])
    ultima_data = str(df["ds"].iloc[-1].date())
    joblib.dump(
        {
            "quantis": modelos_final,
            "features_ultimo_dia": features_hoje,
            "ultimo_fechamento": ultimo_fechamento,
            "ultima_data": ultima_data,
        },
        PASTA_MODELO / "modelo.joblib",
    )

    metadata = {
        "moeda": "EURUSD=X",
        "alvo": "Close",
        "frequencia": "B",
        "treinado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versao_sklearn": sklearn.__version__,
        "metodo": {
            "previsao_central": "último fechamento (passeio aleatório)",
            "intervalo": "ultimo_fechamento * exp(q * sqrt(h)), q = quantil do retorno log de 1 dia, h = dias úteis à frente",
            "quantis": list(QUANTIS),
            "features": list(X.columns),
            "modelo_quantis": "sklearn QuantileRegressor (alpha=0, solver=highs)",
        },
        "avaliacao": {"validacao": validacao, "teste": teste},
        "modelo_final": {
            "periodo": [str(df["ds"].min().date()), ultima_data],
            "linhas_treino": int(linhas.sum()),
            "ultima_data": ultima_data,
            "ultimo_fechamento": ultimo_fechamento,
            "features_ultimo_dia": {k: float(v) for k, v in features_hoje.iloc[0].items()},
            "quantis_retorno_1_dia": {str(q): float(m.predict(features_hoje)[0]) for q, m in modelos_final.items()},
        },
    }
    (PASTA_MODELO / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"\nModelo salvo em {PASTA_MODELO}")


if __name__ == "__main__":
    main()
