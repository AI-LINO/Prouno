import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from hmmlearn import hmm
import requests
from datetime import datetime, timedelta
import time

st.set_page_config(page_title="AI.Lino Quantum Engine", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stSidebar { background-color: #1a1c24; }
    .stMetric { background-color: #262730; padding: 10px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("🚀 Control de análisis AI.Lino")

# ══════════════════════════════════════════════════════════════
#  BINANCE — búsqueda y descarga (sin API key)
# ══════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def binance_get_all_symbols():
    try:
        r = requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=10)
        data = r.json()
        pares = []
        for s in data["symbols"]:
            if s["status"] == "TRADING" and s["quoteAsset"] in ("USDT", "BTC", "ETH", "BNB"):
                pares.append({"symbol": s["symbol"], "base": s["baseAsset"], "quote": s["quoteAsset"]})
        return pares
    except Exception:
        return []

def binance_buscar(query):
    q = query.upper()
    todos = binance_get_all_symbols()
    prioridad = {"USDT": 0, "BTC": 1, "ETH": 2, "BNB": 3}
    encontrados = [p for p in todos if q in p["base"] or q in p["symbol"]]
    encontrados.sort(key=lambda x: prioridad.get(x["quote"], 9))
    return encontrados[:8]

def binance_descargar(symbol, periodo="2y"):
    meses_map = {"1y": 365, "2y": 730, "5y": 1825}
    dias = meses_map.get(periodo, 730)
    start_ms = int((datetime.utcnow() - timedelta(days=dias)).timestamp() * 1000)
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": "1d", "startTime": start_ms, "limit": 1000}
    filas = []
    while True:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if not data or isinstance(data, dict):
            break
        filas.extend(data)
        if len(data) < 1000:
            break
        params["startTime"] = data[-1][0] + 1
    if not filas:
        return pd.DataFrame()
    df = pd.DataFrame(filas, columns=[
        "Open time","Open","High","Low","Close","Volume",
        "Close time","Quote asset volume","Number of trades",
        "Taker buy base","Taker buy quote","Ignore"
    ])
    df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")
    df.set_index("Open time", inplace=True)
    for col in ["Open","High","Low","Close","Volume"]:
        df[col] = df[col].astype(float)
    df.index = df.index.tz_localize("UTC")
    return df[["Open","High","Low","Close","Volume"]]


# ══════════════════════════════════════════════════════════════
#  COINGECKO — búsqueda y descarga (API pública, sin key)
# ══════════════════════════════════════════════════════════════
CG_BASE = "https://api.coingecko.com/api/v3"

def coingecko_buscar(query):
    """Busca coins en CoinGecko por nombre o símbolo."""
    try:
        r = requests.get(f"{CG_BASE}/search", params={"query": query}, timeout=10)
        data = r.json()
        coins = data.get("coins", [])[:8]
        return coins  # cada elemento tiene: id, name, symbol, market_cap_rank, thumb
    except Exception:
        return []

def coingecko_descargar(coin_id, periodo="2y"):
    """Descarga OHLC diario de CoinGecko para un coin_id (ej: 'monad')."""
    dias_map = {"1y": 365, "2y": 730, "5y": 1825}
    dias = dias_map.get(periodo, 730)
    
    # CoinGecko /market_chart devuelve precios de cierre con timestamp
    url = f"{CG_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": dias, "interval": "daily"}
    
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 429:
            time.sleep(60)
            r = requests.get(url, params=params, timeout=15)
        data = r.json()
        
        if "prices" not in data:
            return pd.DataFrame()
        
        # Construir DataFrame con precios (Close) y volumen
        prices = pd.DataFrame(data["prices"], columns=["timestamp", "Close"])
        volumes = pd.DataFrame(data["total_volumes"], columns=["timestamp", "Volume"])
        
        df = prices.merge(volumes, on="timestamp")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        
        # Aproximar OHLC usando el precio de cierre (CoinGecko free solo da Close)
        df["Open"]  = df["Close"].shift(1).fillna(df["Close"])
        df["High"]  = df["Close"]
        df["Low"]   = df["Close"]
        
        return df[["Open","High","Low","Close","Volume"]]
    except Exception as e:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════
#  BUSCADOR UNIFICADO: Yahoo + Binance + CoinGecko
# ══════════════════════════════════════════════════════════════
st.sidebar.subheader("🔍 Buscar Acción / Cripto")

query = st.sidebar.text_input(
    "Escribe nombre o ticker:",
    placeholder="Ej: Monad, PEPE, Tesla, AAPL, Solana...",
    key="search_query"
)

ticker_final  = None
ticker_nombre = None
fuente        = None   # "yahoo" | "binance" | "coingecko"

if query and len(query) >= 2:
    with st.sidebar:
        with st.spinner("Buscando en las 3 fuentes..."):
            # Yahoo Finance
            try:
                res_yf    = yf.Search(query, max_results=4, enable_fuzzy_query=True)
                quotes_yf = res_yf.quotes
            except Exception:
                quotes_yf = []

            # Binance
            quotes_bn = binance_buscar(query)

            # CoinGecko
            quotes_cg = coingecko_buscar(query)

        opciones_display = []
        opciones_map     = {}

        # ── Yahoo ──────────────────────────────────────────────
        if quotes_yf:
            for q in quotes_yf:
                sym   = q.get("symbol", "")
                name  = q.get("longname") or q.get("shortname") or sym
                tipo  = q.get("quoteType", "")
                exch  = q.get("exchange", "")
                label = f"📈 {sym} — {name} [{tipo}] ({exch})  [Yahoo]"
                opciones_display.append(label)
                opciones_map[label] = (sym, name, "yahoo")

        # ── Binance ────────────────────────────────────────────
        if quotes_bn:
            for p in quotes_bn:
                sym   = p["symbol"]
                name  = f"{p['base']} / {p['quote']}"
                label = f"🟡 {sym} — {name}  [Binance]"
                opciones_display.append(label)
                opciones_map[label] = (sym, name, "binance")

        # ── CoinGecko ──────────────────────────────────────────
        if quotes_cg:
            for c in quotes_cg:
                coin_id = c.get("id", "")
                name    = c.get("name", coin_id)
                symbol  = c.get("symbol", "").upper()
                rank    = c.get("market_cap_rank", "?")
                rank_str = f"Rank #{rank}" if rank else "nuevo"
                label   = f"🦎 {symbol} — {name} ({rank_str})  [CoinGecko]"
                opciones_display.append(label)
                opciones_map[label] = (coin_id, f"{name} ({symbol})", "coingecko")

        if opciones_display:
            st.write("**Selecciona el instrumento:**")
            seleccion = st.radio("Resultados:", opciones_display, label_visibility="collapsed")
            if seleccion:
                ticker_final, ticker_nombre, fuente = opciones_map[seleccion]
                iconos = {"yahoo": "📈", "binance": "🟡", "coingecko": "🦎"}
                icono = iconos.get(fuente, "🔵")
                st.success(f"✅ {icono} **{ticker_final}** — {ticker_nombre}")
        else:
            st.warning("No se encontraron resultados. Intenta con otro nombre.")

else:
    # ── Favoritos rápidos ──────────────────────────────────────
    st.sidebar.write("**⭐ Favoritos rápidos:**")
    favoritos = {
        "MON — Monad 🦎":      ("monad",    "Monad (MON)",      "coingecko"),
        "BTC/USDT 🟡":         ("BTCUSDT",  "Bitcoin / USDT",   "binance"),
        "ETH/USDT 🔵":         ("ETHUSDT",  "Ethereum / USDT",  "binance"),
        "SOL/USDT ☀️":         ("SOLUSDT",  "Solana / USDT",    "binance"),
        "PEPE/USDT 🐸":        ("PEPEUSDT", "Pepe / USDT",      "binance"),
        "DOGE/USDT 🐕":        ("DOGEUSDT", "Dogecoin / USDT",  "binance"),
        "NVDA 🟢":             ("NVDA",     "NVIDIA Corp",       "yahoo"),
        "AAPL 🍎":             ("AAPL",     "Apple Inc",         "yahoo"),
        "TSLA ⚡":             ("TSLA",     "Tesla Inc",         "yahoo"),
    }
    fav_sel = st.sidebar.selectbox("O elige un favorito:", list(favoritos.keys()), label_visibility="collapsed")
    ticker_final, ticker_nombre, fuente = favoritos[fav_sel]


# ══════════════════════════════════════════════════════════════
#  CONTROLES
# ══════════════════════════════════════════════════════════════
st.sidebar.divider()

if ticker_final:
    badges = {"yahoo": "📈 Yahoo", "binance": "🟡 Binance", "coingecko": "🦎 CoinGecko"}
    st.sidebar.info(f"📌 **{ticker_final}**  |  {badges.get(fuente, fuente)}")

periodo  = st.sidebar.select_slider("Historial", options=["1y","2y","5y"], value="2y")
ejecutar = st.sidebar.button("📊 EJECUTAR INFERENCIA", use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  FUNCIONES DE ANÁLISIS
# ══════════════════════════════════════════════════════════════
def interpretar_estados(model, X_raw):
    means  = model.means_[:, 0]
    labels = {}
    idx_r  = np.argsort(means)
    labels[idx_r[0]]  = {"nombre": "PÁNICO / BAJISTA",      "color": "#FF4B4B", "desc": "Alta volatilidad y caídas fuertes."}
    labels[idx_r[-1]] = {"nombre": "ALCISTA / EUFORIA",     "color": "#00F000", "desc": "Tendencia positiva sólida."}
    for i in range(model.n_components):
        if i not in labels:
            if means[i] > 0:
                labels[i] = {"nombre": "ACUMULACIÓN",            "color": "#0080FF", "desc": "Crecimiento lento, bajo riesgo."}
            else:
                labels[i] = {"nombre": "LATERAL / DISTRIBUCIÓN", "color": "#FFA500", "desc": "Riesgo de cambio de tendencia."}
    return labels


def cargar_datos(ticker, fuente, periodo):
    if fuente == "binance":
        return binance_descargar(ticker, periodo)
    elif fuente == "coingecko":
        return coingecko_descargar(ticker, periodo)
    else:
        t  = yf.Ticker(ticker)
        df = t.history(period=periodo, interval="1d")
        return df if not df.empty else pd.DataFrame()


# ══════════════════════════════════════════════════════════════
#  EJECUCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════
if ejecutar:
    if not ticker_final:
        st.error("⚠️ Por favor selecciona un instrumento primero.")
        st.stop()

    badges = {"yahoo": "📈 Yahoo Finance", "binance": "🟡 Binance", "coingecko": "🦎 CoinGecko"}
    badge  = badges.get(fuente, fuente)

    with st.spinner(f"Descargando datos de {ticker_final} desde {badge}..."):
        df_raw = cargar_datos(ticker_final, fuente, periodo)

    if df_raw is None or df_raw.empty:
        st.error(
            f"❌ No se pudieron obtener datos para **{ticker_final}**.\n\n"
            "Si usas CoinGecko, espera 1 minuto (límite de peticiones) y vuelve a intentarlo."
        )
        st.stop()

    # Validar que hay suficientes datos para el modelo
    if len(df_raw) < 30:
        st.warning(f"⚠️ Solo {len(df_raw)} días de datos disponibles. Esta cripto es muy nueva — los resultados pueden no ser estables.")

    with st.spinner("Analizando regímenes matemáticos (HMM)..."):
        returns = np.log(df_raw["Close"] / df_raw["Close"].shift(1)).dropna()
        vol     = returns.rolling(min(15, len(returns)//3)).std().dropna()
        data_X  = pd.concat([returns, vol], axis=1).dropna()
        data_X.columns = ["ret", "vol"]
        X_norm  = (data_X - data_X.mean()) / data_X.std()
        X_norm += np.random.normal(0, 1e-6, X_norm.shape)

        best_model, best_bic = None, np.inf
        n_max = min(4, max(2, len(data_X) // 30))  # Adaptar nº de estados a datos disponibles
        for n in range(2, n_max + 1):
            m = hmm.GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000, random_state=42)
            m.fit(X_norm)
            if m.bic(X_norm) < best_bic:
                best_bic   = m.bic(X_norm)
                best_model = m

        states     = best_model.predict(X_norm)
        labels_map = interpretar_estados(best_model, data_X)

    # ── Resultados ──────────────────────────────────────────────
    st.header(f"Análisis Estratégico: {ticker_nombre}")
    st.caption(f"ID: {ticker_final}  |  Fuente: {badge}  |  Período: {periodo}  |  Velas: {len(df_raw)}")

    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.style.use("dark_background")
        close_aligned = df_raw["Close"].loc[data_X.index]
        for i in range(best_model.n_components):
            mask = states == i
            ax.scatter(data_X.index[mask], close_aligned[mask],
                       color=labels_map[i]["color"], label=labels_map[i]["nombre"], s=15)
        ax.set_yscale("log")
        ax.legend()
        ax.set_title(f"Regímenes HMM — {ticker_nombre}", color="white")
        st.pyplot(fig)

    with col2:
        actual = states[-1]
        st.subheader("Estado Actual")
        st.markdown(
            f"<h2 style='color:{labels_map[actual]['color']}'>{labels_map[actual]['nombre']}</h2>",
            unsafe_allow_html=True,
        )
        st.write(f"**Descripción:** {labels_map[actual]['desc']}")
        st.divider()
        st.write("**Probabilidad de Permanencia:**")
        trans_p = best_model.transmat_[actual, actual]
        st.metric("Confianza", f"{trans_p * 100:.1f}%")
        
        # Precio actual
        ultimo_precio = df_raw["Close"].iloc[-1]
        st.divider()
        st.metric("Último precio (USD)", f"${ultimo_precio:,.6f}" if ultimo_precio < 1 else f"${ultimo_precio:,.2f}")

    with st.expander("Ver todos los estados detectados"):
        for i in range(best_model.n_components):
            st.markdown(f"**Estado {i} — {labels_map[i]['nombre']}:** {labels_map[i]['desc']}")
