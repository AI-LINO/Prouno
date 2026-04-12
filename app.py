%%writefile app.py
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from hmmlearn import hmm
import requests
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI.Lino Quantum Engine", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; }
    .main { background-color: #080c14; }
    .stSidebar { background-color: #0d1117; border-right: 1px solid #1a2535; }
    
    .signal-box {
        border-radius: 8px; padding: 16px 20px; margin: 8px 0;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.85rem; border-left: 4px solid;
    }
    .signal-compra  { background:#001a0a; border-color:#00ff88; color:#00ff88; }
    .signal-venta   { background:#1a0005; border-color:#ff3355; color:#ff3355; }
    .signal-neutro  { background:#0a0d1a; border-color:#4488ff; color:#7aabff; }
    .signal-espera  { background:#1a1200; border-color:#ffaa00; color:#ffaa00; }
    
    .metric-card {
        background: #0d1117; border: 1px solid #1a2535;
        border-radius: 8px; padding: 12px 16px; margin: 4px 0;
        font-family: 'Share Tech Mono', monospace;
    }
    .metric-label { color: #4a6080; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: #e0e8ff; font-size: 1.3rem; font-weight: 700; }
    
    .score-bar-wrap { background:#111827; border-radius:6px; height:12px; margin:6px 0; overflow:hidden; }
    .score-bar      { height:100%; border-radius:6px; transition: width 0.5s; }
    
    h1,h2,h3 { font-family: 'Rajdhani', sans-serif !important; font-weight:700 !important; }
    .stButton>button {
        background: linear-gradient(135deg,#0044cc,#0088ff);
        color:white; border:none; border-radius:6px;
        font-family:'Rajdhani',sans-serif; font-weight:700;
        font-size:1rem; letter-spacing:1px;
        padding:0.6rem 1rem; width:100%;
        transition: all 0.2s;
    }
    .stButton>button:hover { background: linear-gradient(135deg,#0055ff,#00aaff); transform:translateY(-1px); }
    </style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  CONFIGURACIÓN DE TIMEFRAMES
# ══════════════════════════════════════════════════════════════
TIMEFRAMES = {
    # label           : (interval_binance, interval_yfinance, days_back, label_display)
    "1H  — 3 días"   : ("1h",  "1h",  3,    "3 días · velas 1h"),
    "4H  — 10 días"  : ("4h",  "1h",  10,   "10 días · velas 4h"),
    "1D  — 1 mes"    : ("1d",  "1d",  30,   "1 mes · velas diarias"),
    "1D  — 3 meses"  : ("1d",  "1d",  90,   "3 meses · velas diarias"),
    "1D  — 6 meses"  : ("1d",  "1d",  180,  "6 meses · velas diarias"),
    "1W  — 1 año"    : ("1d",  "1wk", 365,  "1 año · velas semanales"),
}

# ══════════════════════════════════════════════════════════════
#  FUENTES DE DATOS
# ══════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def binance_get_all_symbols():
    try:
        r = requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=10)
        data = r.json()
        return [{"symbol": s["symbol"], "base": s["baseAsset"], "quote": s["quoteAsset"]}
                for s in data["symbols"]
                if s["status"] == "TRADING" and s["quoteAsset"] in ("USDT","BTC","ETH","BNB")]
    except Exception:
        return []

def binance_buscar(query):
    q = query.upper()
    todos = binance_get_all_symbols()
    prio = {"USDT":0,"BTC":1,"ETH":2,"BNB":3}
    found = [p for p in todos if q in p["base"] or q in p["symbol"]]
    found.sort(key=lambda x: prio.get(x["quote"],9))
    return found[:8]

def binance_descargar(symbol, interval_bn, dias):
    start_ms = int((datetime.utcnow() - timedelta(days=dias)).timestamp() * 1000)
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval_bn, "startTime": start_ms, "limit": 1000}
    filas = []
    for _ in range(5):  # max 5 páginas
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if not data or isinstance(data, dict): break
        filas.extend(data)
        if len(data) < 1000: break
        params["startTime"] = data[-1][0] + 1
    if not filas: return pd.DataFrame()
    df = pd.DataFrame(filas, columns=[
        "Open time","Open","High","Low","Close","Volume",
        "Close time","qav","trades","tbb","tbq","ignore"])
    df["Open time"] = pd.to_datetime(df["Open time"], unit="ms", utc=True)
    df.set_index("Open time", inplace=True)
    for col in ["Open","High","Low","Close","Volume"]:
        df[col] = df[col].astype(float)
    return df[["Open","High","Low","Close","Volume"]]

CG_BASE = "https://api.coingecko.com/api/v3"

def coingecko_buscar(query):
    try:
        r = requests.get(f"{CG_BASE}/search", params={"query": query}, timeout=10)
        return r.json().get("coins", [])[:6]
    except Exception:
        return []

def coingecko_descargar(coin_id, dias):
    url = f"{CG_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": dias, "interval": "daily" if dias > 3 else "hourly"}
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 429:
            time.sleep(30)
            r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if "prices" not in data: return pd.DataFrame()
        prices  = pd.DataFrame(data["prices"],         columns=["ts","Close"])
        volumes = pd.DataFrame(data["total_volumes"],  columns=["ts","Volume"])
        df = prices.merge(volumes, on="ts")
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df.set_index("ts", inplace=True)
        df["Open"]  = df["Close"].shift(1).fillna(df["Close"])
        df["High"]  = df["Close"]
        df["Low"]   = df["Close"]
        return df[["Open","High","Low","Close","Volume"]]
    except Exception:
        return pd.DataFrame()

def yahoo_descargar(ticker, interval_yf, dias):
    periodo_map = {3:"5d", 10:"1mo", 30:"1mo", 90:"3mo", 180:"6mo", 365:"1y"}
    periodo = periodo_map.get(dias, "1y")
    t = yf.Ticker(ticker)
    df = t.history(period=periodo, interval=interval_yf)
    return df if not df.empty else pd.DataFrame()

def cargar_datos(ticker, fuente, tf_key):
    interval_bn, interval_yf, dias, _ = TIMEFRAMES[tf_key]
    if fuente == "binance":
        return binance_descargar(ticker, interval_bn, dias)
    elif fuente == "coingecko":
        return coingecko_descargar(ticker, dias)
    else:
        return yahoo_descargar(ticker, interval_yf, dias)


# ══════════════════════════════════════════════════════════════
#  MATEMÁTICAS — INDICADORES TÉCNICOS
# ══════════════════════════════════════════════════════════════
def calcular_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calcular_macd(close, fast=12, slow=26, signal=9):
    ema_fast   = close.ewm(span=fast,   adjust=False).mean()
    ema_slow   = close.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line= macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram

def calcular_bollinger(close, period=20, std_dev=2):
    sma   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    pct_b = (close - lower) / (upper - lower + 1e-9)  # posición dentro de las bandas 0-1
    return upper, sma, lower, pct_b

def calcular_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calcular_ema(close, span):
    return close.ewm(span=span, adjust=False).mean()

def calcular_stoch_rsi(close, period=14, smooth_k=3, smooth_d=3):
    rsi    = calcular_rsi(close, period)
    min_r  = rsi.rolling(period).min()
    max_r  = rsi.rolling(period).max()
    stoch  = (rsi - min_r) / (max_r - min_r + 1e-9) * 100
    k      = stoch.rolling(smooth_k).mean()
    d      = k.rolling(smooth_d).mean()
    return k, d

def calcular_vwap(df):
    tp   = (df["High"] + df["Low"] + df["Close"]) / 3
    vwap = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()
    return vwap

def calcular_todos_indicadores(df):
    c = df["Close"]
    ind = {}
    ind["rsi"]              = calcular_rsi(c, 14)
    ind["macd"], ind["macd_signal"], ind["macd_hist"] = calcular_macd(c)
    ind["bb_upper"], ind["bb_mid"], ind["bb_lower"], ind["bb_pct"] = calcular_bollinger(c)
    ind["atr"]              = calcular_atr(df["High"], df["Low"], c)
    ind["atr_pct"]          = ind["atr"] / c * 100          # volatilidad relativa %
    ind["ema9"]             = calcular_ema(c, 9)
    ind["ema21"]            = calcular_ema(c, 21)
    ind["ema50"]            = calcular_ema(c, 50)
    ind["ema200"]           = calcular_ema(c, 200)
    ind["stoch_k"], ind["stoch_d"] = calcular_stoch_rsi(c)
    ind["vwap"]             = calcular_vwap(df)
    ind["vol_sma"]          = df["Volume"].rolling(20).mean()
    ind["vol_ratio"]        = df["Volume"] / ind["vol_sma"]  # > 1.5 = volumen alto
    ind["momentum"]         = c.pct_change(5) * 100         # retorno 5 periodos
    ind["retorno_log"]      = np.log(c / c.shift(1))
    ind["volatilidad"]      = ind["retorno_log"].rolling(15).std() * np.sqrt(252) * 100  # vol anualizada %
    return ind


# ══════════════════════════════════════════════════════════════
#  HMM — MODELO OCULTO DE MARKOV
# ══════════════════════════════════════════════════════════════
def entrenar_hmm(df, ind):
    ret = ind["retorno_log"].dropna()
    vol = ind["volatilidad"].dropna()
    mom = ind["momentum"].dropna()
    features = pd.concat([ret, vol, mom], axis=1).dropna()
    features.columns = ["ret","vol","mom"]
    X = (features - features.mean()) / features.std()
    X += np.random.normal(0, 1e-6, X.shape)
    
    best_m, best_bic = None, np.inf
    n_max = min(4, max(2, len(X)//20))
    for n in range(2, n_max+1):
        try:
            m = hmm.GaussianHMM(n_components=n, covariance_type="full", n_iter=2000, random_state=42)
            m.fit(X)
            b = m.bic(X)
            if b < best_bic:
                best_bic, best_m = b, m
        except Exception:
            pass
    
    states = best_m.predict(X)
    return best_m, states, features.index

def etiquetar_hmm(model, df, ind, features_idx):
    means   = model.means_[:, 0]   # media del retorno
    vol_m   = model.means_[:, 1]   # media de volatilidad
    idx_r   = np.argsort(means)
    labels  = {}
    labels[idx_r[0]]  = {"nombre":"PÁNICO / BAJISTA",      "color":"#ff3355", "emoji":"🔴", "señal":"ESPERAR"}
    labels[idx_r[-1]] = {"nombre":"ALCISTA / EUFORIA",     "color":"#00ff88", "emoji":"🟢", "señal":"VIGILAR SALIDA"}
    for i in range(model.n_components):
        if i not in labels:
            if means[i] > 0:
                labels[i] = {"nombre":"ACUMULACIÓN",           "color":"#4488ff", "emoji":"🔵", "señal":"POSIBLE ENTRADA"}
            else:
                labels[i] = {"nombre":"LATERAL / DISTRIBUCIÓN","color":"#ffaa00", "emoji":"🟡", "señal":"CAUTELA"}
    return labels


# ══════════════════════════════════════════════════════════════
#  SCORE COMPUESTO (0–100) — ventaja de trading
# ══════════════════════════════════════════════════════════════
def calcular_score(ind, hmm_estado_label):
    """Genera un score de 0 a 100 que resume la probabilidad de movimiento alcista."""
    puntos = []

    # --- RSI (25 pts) ---
    rsi = ind["rsi"].iloc[-1]
    if pd.isna(rsi): rsi = 50
    if rsi < 30:   puntos.append(("RSI Sobreventa",      25, 25, "🟢"))   # señal fuerte compra
    elif rsi < 45: puntos.append(("RSI Zona baja",       18, 25, "🟡"))
    elif rsi < 55: puntos.append(("RSI Neutral",         12, 25, "⚪"))
    elif rsi < 70: puntos.append(("RSI Zona alta",        7, 25, "🟡"))
    else:          puntos.append(("RSI Sobrecompra",      2, 25, "🔴"))   # señal fuerte venta

    # --- MACD (20 pts) ---
    hist = ind["macd_hist"].iloc[-1]
    prev = ind["macd_hist"].iloc[-2] if len(ind["macd_hist"]) > 1 else hist
    if pd.isna(hist): hist = 0; prev = 0
    cruce_alcista = (prev <= 0 and hist > 0)
    cruce_bajista = (prev >= 0 and hist < 0)
    if cruce_alcista:          puntos.append(("MACD Cruce alcista",   20, 20, "🟢"))
    elif hist > 0 and hist > prev: puntos.append(("MACD Acelerando ↑",15, 20, "🟢"))
    elif hist > 0:             puntos.append(("MACD Positivo",        10, 20, "🟡"))
    elif cruce_bajista:        puntos.append(("MACD Cruce bajista",    0, 20, "🔴"))
    else:                      puntos.append(("MACD Negativo",         4, 20, "🔴"))

    # --- Bollinger %B (15 pts) ---
    pct_b = ind["bb_pct"].iloc[-1]
    if pd.isna(pct_b): pct_b = 0.5
    if pct_b < 0.05:   puntos.append(("BB Banda inferior",   15, 15, "🟢"))  # rebote potencial
    elif pct_b < 0.35: puntos.append(("BB Zona baja",        11, 15, "🟡"))
    elif pct_b < 0.65: puntos.append(("BB Centro",            7, 15, "⚪"))
    elif pct_b < 0.95: puntos.append(("BB Zona alta",         4, 15, "🟡"))
    else:              puntos.append(("BB Banda superior",     1, 15, "🔴"))  # posible reversa

    # --- EMAs (20 pts) ---
    c     = ind["ema9"].index.map(lambda x: x)
    last  = -1
    e9    = ind["ema9"].iloc[last]
    e21   = ind["ema21"].iloc[last]
    e50   = ind["ema50"].iloc[last]
    precio= ind["bb_mid"].index  # proxy; usamos bb_mid como close proxy
    ema_score = 0
    if not pd.isna(e9) and not pd.isna(e21) and not pd.isna(e50):
        if e9 > e21 > e50:  ema_score = 20   # alineación alcista perfecta
        elif e9 > e21:      ema_score = 13
        elif e9 > e50:      ema_score = 8
        else:               ema_score = 3
    puntos.append(("EMAs alineadas", ema_score, 20, "🟢" if ema_score >= 13 else ("🟡" if ema_score >= 8 else "🔴")))

    # --- Volumen (10 pts) ---
    vol_r = ind["vol_ratio"].iloc[-1]
    if pd.isna(vol_r): vol_r = 1.0
    if vol_r > 2.0:    puntos.append(("Volumen muy alto",  10, 10, "🟢"))
    elif vol_r > 1.3:  puntos.append(("Volumen elevado",    7, 10, "🟡"))
    elif vol_r > 0.8:  puntos.append(("Volumen normal",     5, 10, "⚪"))
    else:              puntos.append(("Volumen bajo",        2, 10, "🔴"))

    # --- Stoch RSI (10 pts) ---
    sk = ind["stoch_k"].iloc[-1]
    sd = ind["stoch_d"].iloc[-1]
    if not pd.isna(sk) and not pd.isna(sd):
        if sk < 20 and sk > sd:    puntos.append(("StochRSI Cruce alcista", 10, 10, "🟢"))
        elif sk < 30:              puntos.append(("StochRSI Sobreventa",      7, 10, "🟡"))
        elif sk > 80 and sk < sd:  puntos.append(("StochRSI Cruce bajista",   1, 10, "🔴"))
        elif sk > 70:              puntos.append(("StochRSI Sobrecompra",      3, 10, "🟡"))
        else:                      puntos.append(("StochRSI Neutral",          5, 10, "⚪"))
    else:
        puntos.append(("StochRSI N/D", 5, 10, "⚪"))

    total_pts  = sum(p[1] for p in puntos)
    total_max  = sum(p[2] for p in puntos)
    score      = round(total_pts / total_max * 100)
    return score, puntos


def generar_señal(score, ind, hmm_nombre):
    """Devuelve señal de trading, SL y TP sugeridos."""
    precio   = ind["bb_mid"].iloc[-1]   # usaremos el SMA como proxy de precio actual
    atr      = ind["atr"].iloc[-1]
    if pd.isna(atr): atr = precio * 0.02

    if score >= 72:
        señal = "🟢 COMPRA / LONG"
        clase = "signal-compra"
        sl    = precio - 1.5 * atr
        tp1   = precio + 2.0 * atr
        tp2   = precio + 3.5 * atr
        desc  = "Múltiples indicadores alineados alcistas. Considerar entrada con gestión de riesgo."
    elif score >= 55:
        señal = "🟡 VIGILAR — Posible entrada"
        clase = "signal-espera"
        sl    = precio - 2.0 * atr
        tp1   = precio + 1.5 * atr
        tp2   = precio + 2.5 * atr
        desc  = "Señales mixtas con ligero sesgo alcista. Esperar confirmación adicional."
    elif score >= 38:
        señal = "⚪ NEUTRAL — Sin señal clara"
        clase = "signal-neutro"
        sl    = None; tp1 = None; tp2 = None
        desc  = "Mercado en equilibrio. Mejor esperar una dirección definida."
    else:
        señal = "🔴 VENTA / SHORT o SALIR"
        clase = "signal-venta"
        sl    = precio + 1.5 * atr
        tp1   = precio - 2.0 * atr
        tp2   = precio - 3.5 * atr
        desc  = "Indicadores apuntan a debilidad. Considerar reducir exposición o proteger ganancias."

    return señal, clase, sl, tp1, tp2, desc, precio


# ══════════════════════════════════════════════════════════════
#  SIDEBAR — BUSCADOR UNIFICADO
# ══════════════════════════════════════════════════════════════
st.sidebar.title("🚀 AI.Lino")
st.sidebar.caption("Quantum Engine · Multi-Fuente")

st.sidebar.subheader("🔍 Buscar")
query = st.sidebar.text_input("Acción / Cripto:", placeholder="Ej: Monad, BTC, Tesla...", key="sq")

ticker_final = None; ticker_nombre = None; fuente = None

if query and len(query) >= 2:
    with st.sidebar:
        with st.spinner("Buscando..."):
            try:
                res_yf    = yf.Search(query, max_results=3, enable_fuzzy_query=True)
                quotes_yf = res_yf.quotes
            except Exception:
                quotes_yf = []
            quotes_bn = binance_buscar(query)
            quotes_cg = coingecko_buscar(query)

        opts = []; omap = {}
        for q in quotes_yf:
            sym  = q.get("symbol","")
            name = q.get("longname") or q.get("shortname") or sym
            lbl  = f"📈 {sym} — {name} [Yahoo]"
            opts.append(lbl); omap[lbl] = (sym, name, "yahoo")
        for p in quotes_bn:
            lbl = f"🟡 {p['symbol']} — {p['base']}/{p['quote']} [Binance]"
            opts.append(lbl); omap[lbl] = (p["symbol"], f"{p['base']}/{p['quote']}", "binance")
        for c in quotes_cg:
            cid = c.get("id",""); nm = c.get("name",cid); sym = c.get("symbol","").upper()
            rk  = c.get("market_cap_rank","?")
            lbl = f"🦎 {sym} — {nm} (#{rk}) [CoinGecko]"
            opts.append(lbl); omap[lbl] = (cid, f"{nm} ({sym})", "coingecko")

        if opts:
            sel = st.radio("Resultado:", opts, label_visibility="collapsed")
            if sel:
                ticker_final, ticker_nombre, fuente = omap[sel]
                st.success(f"✅ **{ticker_final}**")
        else:
            st.warning("Sin resultados.")
else:
    st.sidebar.write("**⭐ Favoritos**")
    favs = {
        "MON · Monad 🦎":  ("monad",    "Monad (MON)",     "coingecko"),
        "BTC/USDT 🟡":     ("BTCUSDT",  "Bitcoin/USDT",    "binance"),
        "ETH/USDT 🔵":     ("ETHUSDT",  "Ethereum/USDT",   "binance"),
        "SOL/USDT ☀️":     ("SOLUSDT",  "Solana/USDT",     "binance"),
        "PEPE/USDT 🐸":    ("PEPEUSDT", "Pepe/USDT",       "binance"),
        "DOGE/USDT 🐕":    ("DOGEUSDT", "Dogecoin/USDT",   "binance"),
        "NVDA 🟢":         ("NVDA",     "NVIDIA Corp",      "yahoo"),
        "AAPL 🍎":         ("AAPL",     "Apple Inc",        "yahoo"),
        "TSLA ⚡":         ("TSLA",     "Tesla Inc",        "yahoo"),
        "SPY 📊":          ("SPY",      "S&P 500 ETF",      "yahoo"),
    }
    fav_sel = st.sidebar.selectbox("", list(favs.keys()), label_visibility="collapsed")
    ticker_final, ticker_nombre, fuente = favs[fav_sel]

st.sidebar.divider()

# Timeframe selector
tf_key = st.sidebar.selectbox("⏱ Timeframe", list(TIMEFRAMES.keys()), index=2)

if ticker_final:
    bdg = {"yahoo":"📈 Yahoo","binance":"🟡 Binance","coingecko":"🦎 CoinGecko"}.get(fuente,"")
    st.sidebar.info(f"📌 **{ticker_final}** · {bdg}")

ejecutar = st.sidebar.button("⚡ ANALIZAR AHORA", use_container_width=True)

st.sidebar.divider()
st.sidebar.caption("⚠️ Solo educativo. No es asesoría financiera.")


# ══════════════════════════════════════════════════════════════
#  EJECUCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════
if ejecutar:
    if not ticker_final:
        st.error("Selecciona un instrumento primero.")
        st.stop()

    bdg = {"yahoo":"📈 Yahoo Finance","binance":"🟡 Binance","coingecko":"🦎 CoinGecko"}.get(fuente,"")
    _, _, dias, tf_display = TIMEFRAMES[tf_key]

    # ── Descarga de datos ────────────────────────────────────
    with st.spinner(f"⬇️ Descargando {ticker_final} desde {bdg}..."):
        df_raw = cargar_datos(ticker_final, fuente, tf_key)

    if df_raw is None or df_raw.empty or len(df_raw) < 20:
        st.error(f"❌ No hay suficientes datos para **{ticker_final}** en este timeframe. Prueba un período mayor.")
        st.stop()

    # ── Indicadores técnicos ─────────────────────────────────
    with st.spinner("🧮 Calculando indicadores técnicos..."):
        ind = calcular_todos_indicadores(df_raw)

    # ── HMM ─────────────────────────────────────────────────
    with st.spinner("🤖 Entrenando modelo HMM..."):
        try:
            best_model, states, feat_idx = entrenar_hmm(df_raw, ind)
            labels_map = etiquetar_hmm(best_model, df_raw, ind, feat_idx)
            hmm_ok = True
        except Exception as e:
            hmm_ok = False

    # ── Score y señal ─────────────────────────────────────────
    hmm_nombre = labels_map[states[-1]]["nombre"] if hmm_ok else "N/D"
    score, desglose = calcular_score(ind, hmm_nombre)
    señal, clase_señal, sl, tp1, tp2, desc_señal, precio_ref = generar_señal(score, ind, hmm_nombre)

    precio_actual = df_raw["Close"].iloc[-1]
    cambio_1p     = df_raw["Close"].pct_change(1).iloc[-1] * 100
    vol_actual    = ind["atr_pct"].iloc[-1]

    # ══════════════════════════════════════════════════════════
    #  UI — HEADER
    # ══════════════════════════════════════════════════════════
    st.markdown(f"## {ticker_nombre}")
    st.caption(f"{bdg}  ·  {tf_display}  ·  {len(df_raw)} velas  ·  Actualizado: {datetime.utcnow().strftime('%H:%M UTC')}")

    # ── Fila 1: métricas rápidas ─────────────────────────────
    c1,c2,c3,c4,c5 = st.columns(5)
    precio_fmt = f"${precio_actual:,.6f}" if precio_actual < 1 else f"${precio_actual:,.4f}" if precio_actual < 10 else f"${precio_actual:,.2f}"
    c1.metric("Precio",      precio_fmt, f"{cambio_1p:+.2f}%")
    c2.metric("RSI (14)",    f"{ind['rsi'].iloc[-1]:.1f}" if not pd.isna(ind['rsi'].iloc[-1]) else "N/D")
    c3.metric("Volatilidad", f"{vol_actual:.1f}%" if not pd.isna(vol_actual) else "N/D")
    c4.metric("Vol. Ratio",  f"{ind['vol_ratio'].iloc[-1]:.2f}x" if not pd.isna(ind['vol_ratio'].iloc[-1]) else "N/D")
    c5.metric("Score IA",    f"{score}/100")

    st.divider()

    # ── Fila 2: Gráfico + Panel derecho ──────────────────────
    col_chart, col_panel = st.columns([3, 1])

    with col_chart:
        plt.style.use("dark_background")
        fig = plt.figure(figsize=(13, 10), facecolor="#080c14")
        gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.08,
                                height_ratios=[3, 1, 1, 1])

        ax1 = fig.add_subplot(gs[0])  # Precio + HMM + EMAs + BB
        ax2 = fig.add_subplot(gs[1], sharex=ax1)  # Volumen
        ax3 = fig.add_subplot(gs[2], sharex=ax1)  # RSI + StochRSI
        ax4 = fig.add_subplot(gs[3], sharex=ax1)  # MACD

        idx = df_raw.index
        c   = df_raw["Close"]

        # --- ax1: Precio con colores HMM ---
        if hmm_ok:
            close_feat = df_raw["Close"].loc[feat_idx]
            for i in range(best_model.n_components):
                mask = states == i
                ax1.scatter(feat_idx[mask], close_feat[mask],
                            color=labels_map[i]["color"], s=12, alpha=0.7, zorder=3)
        ax1.plot(idx, c, color="#334466", linewidth=0.8, alpha=0.5, zorder=2)
        ax1.plot(idx, ind["ema9"],   color="#ffdd44", linewidth=1,   alpha=0.8, label="EMA9")
        ax1.plot(idx, ind["ema21"],  color="#ff8844", linewidth=1,   alpha=0.8, label="EMA21")
        ax1.plot(idx, ind["ema50"],  color="#cc44ff", linewidth=1.2, alpha=0.9, label="EMA50")
        ax1.fill_between(idx, ind["bb_upper"], ind["bb_lower"],
                         alpha=0.06, color="#4488ff", label="Bollinger")
        ax1.plot(idx, ind["bb_upper"], color="#2255aa", linewidth=0.7, linestyle="--")
        ax1.plot(idx, ind["bb_lower"], color="#2255aa", linewidth=0.7, linestyle="--")
        if fuente != "coingecko":
            ax1.plot(idx, ind["vwap"], color="#00ffcc", linewidth=0.9, linestyle=":", alpha=0.7, label="VWAP")
        ax1.set_ylabel("Precio", color="#4a6080", fontsize=9)
        ax1.legend(loc="upper left", fontsize=7, framealpha=0.3)
        ax1.set_facecolor("#080c14")
        ax1.tick_params(labelbottom=False, colors="#4a6080")
        for sp in ax1.spines.values(): sp.set_color("#1a2535")

        # --- ax2: Volumen con color por dirección ---
        colores_vol = ["#00ff88" if df_raw["Close"].iloc[i] >= df_raw["Open"].iloc[i] else "#ff3355"
                       for i in range(len(df_raw))]
        ax2.bar(idx, df_raw["Volume"], color=colores_vol, alpha=0.6, width=0.8)
        ax2.plot(idx, ind["vol_sma"] if "vol_sma" in ind else df_raw["Volume"].rolling(20).mean(),
                 color="#ffaa00", linewidth=1, alpha=0.8)
        ax2.set_ylabel("Vol", color="#4a6080", fontsize=8)
        ax2.set_facecolor("#080c14")
        ax2.tick_params(labelbottom=False, colors="#4a6080")
        for sp in ax2.spines.values(): sp.set_color("#1a2535")

        # --- ax3: RSI + zonas ---
        ax3.plot(idx, ind["rsi"], color="#ff8844", linewidth=1.2, label="RSI14")
        ax3.plot(idx, ind["stoch_k"], color="#44aaff", linewidth=0.9, alpha=0.7, label="StochK")
        ax3.axhline(70, color="#ff3355", linewidth=0.7, linestyle="--", alpha=0.6)
        ax3.axhline(30, color="#00ff88", linewidth=0.7, linestyle="--", alpha=0.6)
        ax3.axhline(50, color="#334466", linewidth=0.5, linestyle=":")
        ax3.fill_between(idx, 70, 100, alpha=0.05, color="#ff3355")
        ax3.fill_between(idx, 0,   30, alpha=0.05, color="#00ff88")
        ax3.set_ylim(0, 100)
        ax3.set_ylabel("RSI", color="#4a6080", fontsize=8)
        ax3.legend(loc="upper left", fontsize=7, framealpha=0.3)
        ax3.set_facecolor("#080c14")
        ax3.tick_params(labelbottom=False, colors="#4a6080")
        for sp in ax3.spines.values(): sp.set_color("#1a2535")

        # --- ax4: MACD ---
        colors_hist = ["#00ff88" if v >= 0 else "#ff3355" for v in ind["macd_hist"]]
        ax4.bar(idx, ind["macd_hist"], color=colors_hist, alpha=0.7, width=0.8)
        ax4.plot(idx, ind["macd"],        color="#4488ff", linewidth=1.2, label="MACD")
        ax4.plot(idx, ind["macd_signal"], color="#ffaa00", linewidth=1,   label="Signal")
        ax4.axhline(0, color="#334466", linewidth=0.5, linestyle=":")
        ax4.set_ylabel("MACD", color="#4a6080", fontsize=8)
        ax4.legend(loc="upper left", fontsize=7, framealpha=0.3)
        ax4.set_facecolor("#080c14")
        ax4.tick_params(colors="#4a6080", labelrotation=30, labelsize=7)
        for sp in ax4.spines.values(): sp.set_color("#1a2535")

        plt.setp(ax1.xaxis.get_majorticklabels(), visible=False)
        plt.setp(ax2.xaxis.get_majorticklabels(), visible=False)
        plt.setp(ax3.xaxis.get_majorticklabels(), visible=False)

        st.pyplot(fig)

    # ── Panel derecho: señal, score, métricas ────────────────
    with col_panel:

        # SEÑAL PRINCIPAL
        st.markdown(f"""
        <div class="signal-box {clase_señal}">
            <div style="font-size:1.1rem; font-weight:700; margin-bottom:6px">{señal}</div>
            <div style="font-size:0.78rem; opacity:0.8">{desc_señal}</div>
        </div>""", unsafe_allow_html=True)

        # HMM Régimen
        if hmm_ok:
            estado_actual = labels_map[states[-1]]
            st.markdown(f"""
            <div class="metric-card" style="border-left:3px solid {estado_actual['color']}">
                <div class="metric-label">Régimen HMM</div>
                <div class="metric-value" style="color:{estado_actual['color']};font-size:1rem">
                    {estado_actual['emoji']} {estado_actual['nombre']}
                </div>
                <div style="color:#4a6080;font-size:0.75rem;margin-top:4px">
                    Permanencia: {best_model.transmat_[states[-1], states[-1]]*100:.0f}%
                </div>
            </div>""", unsafe_allow_html=True)

        # SCORE BAR
        score_color = "#00ff88" if score >= 65 else ("#ffaa00" if score >= 45 else "#ff3355")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Score de Trading</div>
            <div class="metric-value" style="color:{score_color}">{score} / 100</div>
            <div class="score-bar-wrap">
                <div class="score-bar" style="width:{score}%;background:{score_color}"></div>
            </div>
        </div>""", unsafe_allow_html=True)

        # SL / TP
        if sl is not None:
            def fmt_p(v):
                if v is None: return "—"
                return f"${v:,.6f}" if v < 1 else f"${v:,.4f}" if v < 10 else f"${v:,.2f}"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Niveles sugeridos (ATR×)</div>
                <div style="color:#ff3355;font-size:0.85rem">🛑 Stop Loss:  {fmt_p(sl)}</div>
                <div style="color:#ffaa00;font-size:0.85rem">🎯 TP1:  {fmt_p(tp1)}</div>
                <div style="color:#00ff88;font-size:0.85rem">🎯 TP2:  {fmt_p(tp2)}</div>
                <div style="color:#4a6080;font-size:0.72rem;margin-top:4px">Basado en ATR-14</div>
            </div>""", unsafe_allow_html=True)

        # DESGLOSE DEL SCORE
        st.markdown("**Desglose del Score:**")
        for nombre, pts, max_pts, emoji in desglose:
            pct = pts / max_pts * 100
            color = "#00ff88" if pct >= 70 else ("#ffaa00" if pct >= 40 else "#ff3355")
            st.markdown(f"""
            <div style="margin:3px 0">
                <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#8899aa">
                    <span>{emoji} {nombre}</span>
                    <span style="color:{color}">{pts}/{max_pts}</span>
                </div>
                <div class="score-bar-wrap" style="height:6px">
                    <div class="score-bar" style="width:{pct:.0f}%;background:{color}"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    #  TABLA DE INDICADORES COMPLETA
    # ══════════════════════════════════════════════════════════
    st.divider()
    with st.expander("📊 Tabla completa de indicadores", expanded=False):
        rsi_v    = ind["rsi"].iloc[-1]
        macd_v   = ind["macd"].iloc[-1]
        msig_v   = ind["macd_signal"].iloc[-1]
        mhist_v  = ind["macd_hist"].iloc[-1]
        bb_u     = ind["bb_upper"].iloc[-1]
        bb_l     = ind["bb_lower"].iloc[-1]
        bb_m     = ind["bb_mid"].iloc[-1]
        atr_v    = ind["atr"].iloc[-1]
        atr_pct  = ind["atr_pct"].iloc[-1]
        sk, sd   = ind["stoch_k"].iloc[-1], ind["stoch_d"].iloc[-1]
        e9,e21,e50,e200 = ind["ema9"].iloc[-1],ind["ema21"].iloc[-1],ind["ema50"].iloc[-1],ind["ema200"].iloc[-1]
        vr       = ind["vol_ratio"].iloc[-1]
        mom      = ind["momentum"].iloc[-1]

        def fv(v, dec=4):
            if pd.isna(v): return "N/D"
            return f"{v:.{dec}f}"

        tabla = {
            "Indicador": ["RSI(14)","MACD","MACD Signal","MACD Hist",
                          "BB Superior","BB Medio","BB Inferior","BB %B",
                          "ATR(14)","ATR %","EMA9","EMA21","EMA50","EMA200",
                          "StochRSI K","StochRSI D","Vol Ratio","Momentum 5p"],
            "Valor": [
                fv(rsi_v,1), fv(macd_v,6), fv(msig_v,6), fv(mhist_v,6),
                fv(bb_u,4), fv(bb_m,4), fv(bb_l,4), fv(ind["bb_pct"].iloc[-1],3),
                fv(atr_v,6), f"{fv(atr_pct,2)}%",
                fv(e9,4), fv(e21,4), fv(e50,4), fv(e200,4),
                fv(sk,1), fv(sd,1), f"{fv(vr,2)}x", f"{fv(mom,2)}%"
            ],
            "Interpretación": [
                "Sobrevendido<30, Sobrecomprado>70",
                "Positivo = impulso alcista",
                "MACD > Signal = alcista",
                "Histograma > 0 = momentum alcista",
                "Resistencia dinámica",
                "Media móvil 20",
                "Soporte dinámico",
                "<0.2 sobrevendido, >0.8 sobrecomprado",
                "Volatilidad absoluta promedio",
                "% volatilidad relativa al precio",
                "Promedio corto plazo",
                "Promedio medio plazo",
                "Promedio largo plazo",
                "Tendencia anual",
                "< 20 zona compra, > 80 zona venta",
                "Confirmación del K",
                "> 1.5x = volumen significativo",
                "Retorno últimos 5 períodos"
            ]
        }
        st.dataframe(pd.DataFrame(tabla), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════
    #  GLOSARIO HMM
    # ══════════════════════════════════════════════════════════
    if hmm_ok:
        with st.expander("🤖 Estados del modelo HMM detectados"):
            for i in range(best_model.n_components):
                lm = labels_map[i]
                trans = best_model.transmat_[i, i] * 100
                st.markdown(
                    f"**{lm['emoji']} Estado {i} — {lm['nombre']}** "
                    f"· Acción sugerida: *{lm['señal']}* "
                    f"· Prob. permanencia: **{trans:.0f}%**"
                )
