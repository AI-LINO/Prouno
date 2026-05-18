# ╔══════════════════════════════════════════════════════════════╗
# ║          AI.LINO QUANTUM ENGINE v4.0                        ║
# ║  Filtro Tendencia · Entradas Precisas · Trailing · MTF     ║
# ║  Backtesting · Confluence Score · HMM 5D                   ║
# ╚══════════════════════════════════════════════════════════════╝
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from hmmlearn import hmm
import requests
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI.Lino v4", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700&family=Rajdhani:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Rajdhani',sans-serif}
.main{background-color:#050810}.stSidebar{background-color:#08090f;border-right:1px solid #0d1a2e}
h1,h2,h3{font-family:'Orbitron',monospace !important;letter-spacing:2px}
.qcard{background:linear-gradient(135deg,#080c18,#0d1428);border:1px solid #1a2a4a;
       border-radius:10px;padding:14px 18px;margin:8px 0;font-family:'Share Tech Mono',monospace}
.qtitle{font-family:'Orbitron',monospace;font-size:0.82rem;color:#4488ff;letter-spacing:3px;
        text-transform:uppercase;margin-bottom:8px;border-bottom:1px solid #1a2a4a;padding-bottom:6px}
.sbox{border-radius:8px;padding:12px 16px;margin:6px 0;
      font-family:'Share Tech Mono',monospace;font-size:0.82rem;border-left:4px solid}
.sc{background:#001a0a;border-color:#00ff88;color:#00ff88}
.sv{background:#1a0005;border-color:#ff3355;color:#ff3355}
.sn{background:#0a0d1a;border-color:#4488ff;color:#7aabff}
.se{background:#1a1200;border-color:#ffaa00;color:#ffaa00}
.mc{background:#08090f;border:1px solid #1a2a4a;border-radius:8px;padding:9px 13px;margin:3px 0;
    font-family:'Share Tech Mono',monospace}
.ml{color:#2a4060;font-size:0.68rem;text-transform:uppercase;letter-spacing:1px}
.mv{color:#c0d8ff;font-size:1.15rem;font-weight:700}
.bw{background:#0d1428;border-radius:4px;height:9px;margin:4px 0;overflow:hidden}
.bf{height:100%;border-radius:4px}
.stButton>button{background:linear-gradient(135deg,#001166,#0044cc,#0088ff);
 color:#aaddff;border:none;border-radius:6px;font-family:'Orbitron',monospace;
 font-weight:700;font-size:0.78rem;letter-spacing:2px;padding:0.65rem 1rem;
 width:100%;transition:all 0.3s}
.stButton>button:hover{background:linear-gradient(135deg,#0022aa,#0066ff,#00aaff);
 color:white;transform:translateY(-2px);box-shadow:0 4px 20px #0044ff44}
.trend-alcista{background:#001a0a;border:1px solid #00ff88;border-radius:8px;
               padding:10px 14px;font-family:'Share Tech Mono',monospace;color:#00ff88}
.trend-bajista{background:#1a0005;border:1px solid #ff3355;border-radius:8px;
               padding:10px 14px;font-family:'Share Tech Mono',monospace;color:#ff3355}
.trend-neutral{background:#0a0d1a;border:1px solid #4488ff;border-radius:8px;
               padding:10px 14px;font-family:'Share Tech Mono',monospace;color:#7aabff}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  TIMEFRAMES
# ══════════════════════════════════════════════════════════════
TIMEFRAMES = {
    "1H  · 3 días"  : ("1h",  "1h",   3,   "3d·1H"),
    "4H  · 10 días" : ("4h",  "1h",   10,  "10d·4H"),
    "1D  · 1 mes"   : ("1d",  "1d",   30,  "1m·1D"),
    "1D  · 3 meses" : ("1d",  "1d",   90,  "3m·1D"),
    "1D  · 6 meses" : ("1d",  "1d",   180, "6m·1D"),
    "1W  · 1 año"   : ("1d",  "1wk",  365, "1a·1W"),
}
# Mapa de timeframe superior para MTF
MTF_SUPERIOR = {
    "1H  · 3 días"  : ("1d",  "1d",   30),
    "4H  · 10 días" : ("1d",  "1d",   90),
    "1D  · 1 mes"   : ("1d",  "1wk",  365),
    "1D  · 3 meses" : ("1d",  "1wk",  365),
    "1D  · 6 meses" : ("1d",  "1wk",  365),
    "1W  · 1 año"   : ("1d",  "1wk",  365),
}

BG = "#050810"

# ══════════════════════════════════════════════════════════════
#  FUENTES DE DATOS
# ══════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def binance_get_all_symbols():
    try:
        r = requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=10)
        return [{"symbol":s["symbol"],"base":s["baseAsset"],"quote":s["quoteAsset"]}
                for s in r.json()["symbols"]
                if s["status"]=="TRADING" and s["quoteAsset"] in ("USDT","BTC","ETH","BNB")]
    except: return []

def binance_buscar(q):
    q = q.upper(); todos = binance_get_all_symbols()
    prio = {"USDT":0,"BTC":1,"ETH":2,"BNB":3}
    found = [p for p in todos if q in p["base"] or q in p["symbol"]]
    found.sort(key=lambda x: prio.get(x["quote"],9))
    return found[:8]

def binance_descargar(symbol, interval_bn, dias):
    start_ms = int((datetime.utcnow()-timedelta(days=dias)).timestamp()*1000)
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol":symbol,"interval":interval_bn,"startTime":start_ms,"limit":1000}
    filas = []
    for _ in range(5):
        try:
            r = requests.get(url, params=params, timeout=15); data = r.json()
            if not data or isinstance(data,dict): break
            filas.extend(data)
            if len(data)<1000: break
            params["startTime"] = data[-1][0]+1
        except: break
    if not filas: return pd.DataFrame()
    df = pd.DataFrame(filas, columns=["Open time","Open","High","Low","Close","Volume",
                                       "Close time","qav","trades","tbb","tbq","ignore"])
    df["Open time"] = pd.to_datetime(df["Open time"], unit="ms", utc=True)
    df.set_index("Open time", inplace=True)
    for col in ["Open","High","Low","Close","Volume"]: df[col] = df[col].astype(float)
    return df[["Open","High","Low","Close","Volume"]]

CG_BASE = "https://api.coingecko.com/api/v3"

def coingecko_buscar(query):
    try:
        r = requests.get(f"{CG_BASE}/search", params={"query":query}, timeout=10)
        return r.json().get("coins",[])[:6]
    except: return []

def coingecko_descargar(coin_id, dias):
    url = f"{CG_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency":"usd","days":dias,"interval":"daily" if dias>3 else "hourly"}
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code==429: time.sleep(30); r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if "prices" not in data: return pd.DataFrame()
        prices  = pd.DataFrame(data["prices"],        columns=["ts","Close"])
        volumes = pd.DataFrame(data["total_volumes"],  columns=["ts","Volume"])
        df = prices.merge(volumes, on="ts")
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df.set_index("ts", inplace=True)
        df["Open"] = df["Close"].shift(1).fillna(df["Close"])
        df["High"] = df["Close"]; df["Low"] = df["Close"]
        return df[["Open","High","Low","Close","Volume"]]
    except: return pd.DataFrame()

def yahoo_descargar(ticker, interval_yf, dias):
    pm = {3:"5d",10:"1mo",30:"1mo",90:"3mo",180:"6mo",365:"1y"}
    try:
        t = yf.Ticker(ticker); df = t.history(period=pm.get(dias,"1y"), interval=interval_yf)
        return df if not df.empty else pd.DataFrame()
    except: return pd.DataFrame()

def cargar_datos(ticker, fuente, tf_key):
    ibn,iyf,dias,_ = TIMEFRAMES[tf_key]
    if fuente=="binance":    return binance_descargar(ticker, ibn, dias)
    elif fuente=="coingecko": return coingecko_descargar(ticker, dias)
    else:                    return yahoo_descargar(ticker, iyf, dias)

def cargar_datos_superior(ticker, fuente, tf_key):
    """Descarga el timeframe superior para confirmación MTF."""
    ibn_s, iyf_s, dias_s = MTF_SUPERIOR[tf_key]
    if fuente=="binance":    return binance_descargar(ticker, ibn_s, dias_s)
    elif fuente=="coingecko": return coingecko_descargar(ticker, dias_s)
    else:                    return yahoo_descargar(ticker, iyf_s, dias_s)

def fp(v):
    if v is None or (isinstance(v,float) and np.isnan(v)): return "—"
    return f"${v:,.6f}" if abs(v)<1 else f"${v:,.4f}" if abs(v)<10 else f"${v:,.2f}"

# ══════════════════════════════════════════════════════════════
#  INDICADORES TÉCNICOS
# ══════════════════════════════════════════════════════════════
def calcular_rsi(c, p=14):
    d = c.diff()
    g = d.clip(lower=0).ewm(com=p-1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=p-1, adjust=False).mean()
    return 100-(100/(1+g/(l+1e-10)))

def calcular_macd(c, f=12, s=26, sig=9):
    ef = c.ewm(span=f, adjust=False).mean()
    es = c.ewm(span=s, adjust=False).mean()
    m = ef-es; sl = m.ewm(span=sig, adjust=False).mean()
    return m, sl, m-sl

def calcular_bb(c, p=20, k=2):
    sma = c.rolling(p).mean(); std = c.rolling(p).std()
    up = sma+k*std; lo = sma-k*std
    return up, sma, lo, (c-lo)/(up-lo+1e-9)

def calcular_atr(h, l, c, p=14):
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()

def calcular_stoch_rsi(c, p=14, sk=3, sd=3):
    rsi = calcular_rsi(c,p)
    mn = rsi.rolling(p).min(); mx = rsi.rolling(p).max()
    st = (rsi-mn)/(mx-mn+1e-9)*100
    k = st.rolling(sk).mean()
    return k, k.rolling(sd).mean()

def calcular_indicadores(df):
    c = df["Close"]; ind = {}
    ind["rsi"]  = calcular_rsi(c)
    ind["macd"],ind["macd_sig"],ind["macd_hist"] = calcular_macd(c)
    ind["bb_up"],ind["bb_mid"],ind["bb_lo"],ind["bb_pct"] = calcular_bb(c)
    ind["atr"]  = calcular_atr(df["High"], df["Low"], c)
    ind["atr_pct"] = ind["atr"]/c*100
    ind["ema9"]  = c.ewm(span=9,  adjust=False).mean()
    ind["ema21"] = c.ewm(span=21, adjust=False).mean()
    ind["ema50"] = c.ewm(span=50, adjust=False).mean()
    ind["ema200"]= c.ewm(span=200,adjust=False).mean()
    ind["stoch_k"],ind["stoch_d"] = calcular_stoch_rsi(c)
    tp = (df["High"]+df["Low"]+c)/3
    ind["vwap"]    = (tp*df["Volume"]).cumsum()/df["Volume"].cumsum()
    ind["vol_sma"] = df["Volume"].rolling(20).mean()
    ind["vol_ratio"]= df["Volume"]/(ind["vol_sma"]+1e-10)
    ind["momentum"] = c.pct_change(5)*100
    ind["ret_log"]  = np.log(c/(c.shift(1)+1e-10))
    ind["vol_ann"]  = ind["ret_log"].rolling(15).std()*np.sqrt(252)*100
    ind["chg1"] = c.pct_change(1)*100
    ind["chg3"] = c.pct_change(3)*100
    return ind

# ══════════════════════════════════════════════════════════════
#  FILTRO DE TENDENCIA (mejora #1 de Grok)
# ══════════════════════════════════════════════════════════════
def filtro_tendencia(df, ind):
    """
    Clasificación robusta de tendencia usando EMA200 + alineación de EMAs.
    Retorna: tendencia, fuerza (0-100), ema200, estado para bloquear señales contrarias.
    """
    c    = df["Close"]
    e9   = ind["ema9"].iloc[-1]
    e21  = ind["ema21"].iloc[-1]
    e50  = ind["ema50"].iloc[-1]
    e200 = ind["ema200"].iloc[-1]
    close= c.iloc[-1]

    # Pendiente EMA50 (últimas 5 velas)
    e50_series = ind["ema50"]
    pendiente  = (e50_series.iloc[-1] - e50_series.iloc[-6]) / (e50_series.iloc[-6]+1e-10)*100 if len(e50_series)>6 else 0

    # Determinar tendencia
    if close > e200 and e50 > e200:
        if e9 > e21 > e50 > e200 and pendiente > 0.1:
            tend = "ALCISTA FUERTE"; fuerza = 90; clase = "trend-alcista"
        elif e9 > e21 and close > e50:
            tend = "ALCISTA";        fuerza = 70; clase = "trend-alcista"
        else:
            tend = "ALCISTA DÉBIL";  fuerza = 50; clase = "trend-neutral"
    elif close < e200 and e50 < e200:
        if e9 < e21 < e50 < e200 and pendiente < -0.1:
            tend = "BAJISTA FUERTE"; fuerza = 10; clase = "trend-bajista"
        elif e9 < e21 and close < e50:
            tend = "BAJISTA";        fuerza = 30; clase = "trend-bajista"
        else:
            tend = "BAJISTA DÉBIL";  fuerza = 45; clase = "trend-bajista"
    else:
        tend = "LATERAL / SIN TENDENCIA"; fuerza = 50; clase = "trend-neutral"

    fuerte_alcista = "ALCISTA" in tend and fuerza >= 70
    fuerte_bajista = "BAJISTA" in tend and fuerza <= 30

    return {
        "tendencia": tend, "fuerza": fuerza, "clase": clase,
        "fuerte_alcista": fuerte_alcista, "fuerte_bajista": fuerte_bajista,
        "ema200": e200, "pendiente_e50": pendiente,
        "bloquear_long":  fuerte_bajista,
        "bloquear_short": fuerte_alcista,
    }

# ══════════════════════════════════════════════════════════════
#  CONFIRMACIÓN MULTI-TIMEFRAME (MTF) — mejora #4 de Grok
# ══════════════════════════════════════════════════════════════
def analisis_mtf(df_sup):
    """Análisis del timeframe superior para confirmar dirección."""
    if df_sup is None or df_sup.empty or len(df_sup) < 20:
        return {"ok": False, "tendencia": "N/D", "score_sup": 50, "ema50_sup": None}
    ind_s  = calcular_indicadores(df_sup)
    c      = df_sup["Close"]
    e9s    = ind_s["ema9"].iloc[-1]
    e21s   = ind_s["ema21"].iloc[-1]
    e50s   = ind_s["ema50"].iloc[-1]
    e200s  = ind_s["ema200"].iloc[-1]
    rsi_s  = ind_s["rsi"].iloc[-1]
    macd_h = ind_s["macd_hist"].iloc[-1]
    close  = c.iloc[-1]

    if close > e50s > e200s and e9s > e21s:
        tend_sup = "ALCISTA"; score_sup = 75
    elif close < e50s < e200s and e9s < e21s:
        tend_sup = "BAJISTA"; score_sup = 25
    else:
        tend_sup = "LATERAL"; score_sup = 50

    if not pd.isna(rsi_s):
        if rsi_s < 40: score_sup += 10
        elif rsi_s > 65: score_sup -= 10
    if not pd.isna(macd_h) and macd_h > 0: score_sup += 5
    elif not pd.isna(macd_h): score_sup -= 5

    return {"ok":True, "tendencia":tend_sup, "score_sup":int(np.clip(score_sup,0,100)),
            "ema50_sup":e50s, "rsi_sup":rsi_s if not pd.isna(rsi_s) else 50}

# ══════════════════════════════════════════════════════════════
#  HMM MEJORADO — 5 features, múltiples semillas
# ══════════════════════════════════════════════════════════════
def entrenar_hmm(df, ind):
    ret  = ind["ret_log"].dropna()
    vol  = ind["vol_ann"].dropna()
    mom  = ind["momentum"].dropna()
    rsi  = (ind["rsi"].dropna()/100.0)
    pctb = ind["bb_pct"].clip(0,1).dropna()
    feat = pd.concat([ret,vol,mom,rsi,pctb], axis=1).dropna()
    feat.columns = ["r","v","m","rsi","pctb"]
    if len(feat) < 25: feat = feat[["r","v"]]

    med = feat.median(); iqr = feat.quantile(0.75)-feat.quantile(0.25)+1e-8
    X   = ((feat-med)/iqr).values + np.random.normal(0,1e-5,feat.shape)

    best_m, best_score = None, np.inf
    n_max = min(4, max(2, len(X)//20))
    for n in range(2, n_max+1):
        for seed in [42,7,99,13]:
            try:
                m = hmm.GaussianHMM(n_components=n, covariance_type="full",
                                    n_iter=3000, tol=1e-5, random_state=seed)
                m.fit(X)
                states_t = m.predict(X)
                min_frac = np.min(np.bincount(states_t))/len(X)
                penalty  = 0 if min_frac>0.07 else 1e6
                score    = m.bic(X)+penalty
                if score < best_score: best_score, best_m = score, m
            except: pass

    if best_m is None: raise ValueError("HMM no convergió")
    states = best_m.predict(X)
    means  = best_m.means_[:,0]
    idx_r  = np.argsort(means); lmap = {}
    lmap[idx_r[0]]  = {"nombre":"PÁNICO/BAJISTA",    "color":"#ff3355","emoji":"🔴","urgencia":4}
    lmap[idx_r[-1]] = {"nombre":"ALCISTA/EUFORIA",   "color":"#00ff88","emoji":"🟢","urgencia":1}
    for i in range(best_m.n_components):
        if i not in lmap:
            lmap[i] = {"nombre":"ACUMULACIÓN",        "color":"#4488ff","emoji":"🔵","urgencia":2} if means[i]>0.001 \
                 else {"nombre":"LATERAL/DISTRIBUCIÓN","color":"#ffaa00","emoji":"🟡","urgencia":3}
    return best_m, states, feat.index, lmap

# ══════════════════════════════════════════════════════════════
#  CONFLUENCE SCORE (mejora #2 de Grok — más exigente)
# ══════════════════════════════════════════════════════════════
def calcular_score(ind, hmm_urgencia, tendencia_info, mtf_info):
    p = []
    rsi  = ind["rsi"].iloc[-1]  if not pd.isna(ind["rsi"].iloc[-1])  else 50
    hist = ind["macd_hist"].iloc[-1]; prev = ind["macd_hist"].iloc[-2] if len(ind["macd_hist"])>1 else hist
    if pd.isna(hist): hist=0; prev=0
    pct  = ind["bb_pct"].iloc[-1] if not pd.isna(ind["bb_pct"].iloc[-1]) else 0.5
    e9   = ind["ema9"].iloc[-1];  e21=ind["ema21"].iloc[-1]; e50=ind["ema50"].iloc[-1]
    vr   = ind["vol_ratio"].iloc[-1] if not pd.isna(ind["vol_ratio"].iloc[-1]) else 1
    sk   = ind["stoch_k"].iloc[-1]; sd = ind["stoch_d"].iloc[-1]
    chg1 = ind["chg1"].iloc[-1] if not pd.isna(ind["chg1"].iloc[-1]) else 0

    # RSI — 20pts (más sensible en extremos)
    if rsi<22:    p.append(("RSI Extremo S.venta",20,20,"🟢"))
    elif rsi<32:  p.append(("RSI Sobreventa",16,20,"🟢"))
    elif rsi<45:  p.append(("RSI Zona baja",11,20,"🟡"))
    elif rsi<55:  p.append(("RSI Neutral",7,20,"⚪"))
    elif rsi<65:  p.append(("RSI Zona alta",4,20,"🟡"))
    elif rsi<76:  p.append(("RSI Sobrecompra",2,20,"🔴"))
    else:         p.append(("RSI Extremo SC",0,20,"🔴"))

    # MACD — 18pts
    if prev<=0 and hist>0:          p.append(("MACD Cruce alcista",18,18,"🟢"))
    elif hist>0 and hist>prev*1.15: p.append(("MACD Acelerando ↑",14,18,"🟢"))
    elif hist>0:                    p.append(("MACD Positivo",9,18,"🟡"))
    elif prev>=0 and hist<0:        p.append(("MACD Cruce bajista",0,18,"🔴"))
    elif hist<prev*1.15:            p.append(("MACD Cayendo ↓",3,18,"🔴"))
    else:                           p.append(("MACD Negativo",5,18,"🔴"))

    # Bollinger %B — 12pts
    if pct<-0.05:  p.append(("BB Bajo banda inf",12,12,"🟢"))
    elif pct<0.2:  p.append(("BB Zona baja",9,12,"🟡"))
    elif pct<0.55: p.append(("BB Centro",6,12,"⚪"))
    elif pct<0.85: p.append(("BB Zona alta",3,12,"🟡"))
    else:          p.append(("BB Sobre banda sup",0,12,"🔴"))

    # EMAs — 18pts (alineación completa)
    es = 0
    if not any(pd.isna(v) for v in [e9,e21,e50]):
        close_v = ind["bb_mid"].iloc[-1]
        if e9>e21>e50 and close_v>e9:   es=18
        elif e9>e21>e50:                 es=14
        elif e9>e21:                     es=9
        elif e9>e50:                     es=5
        else:                            es=1
    p.append(("EMAs alineadas",es,18,"🟢" if es>=14 else("🟡" if es>=7 else "🔴")))

    # Volumen — 12pts (contexto direccional)
    if vr>2.5 and chg1<-1.5:  p.append(("Vol+caída=capitulación",11,12,"🟢"))
    elif vr>2.5 and chg1>1:   p.append(("Vol+subida=impulso",12,12,"🟢"))
    elif vr>1.5:               p.append(("Vol elevado",8,12,"🟡"))
    elif vr>0.8:               p.append(("Vol normal",6,12,"⚪"))
    else:                      p.append(("Vol muy seco",1,12,"🔴"))

    # StochRSI — 10pts
    if not any(pd.isna(v) for v in [sk,sd]):
        if sk<15 and sk>sd:    p.append(("StochRSI cruce alcista",10,10,"🟢"))
        elif sk<25:            p.append(("StochRSI S.venta",7,10,"🟡"))
        elif sk>85 and sk<sd:  p.append(("StochRSI cruce bajista",0,10,"🔴"))
        elif sk>75:            p.append(("StochRSI S.compra",2,10,"🟡"))
        else:                  p.append(("StochRSI Neutral",5,10,"⚪"))
    else: p.append(("StochRSI N/D",5,10,"⚪"))

    total_pts = sum(x[1] for x in p)
    total_max = sum(x[2] for x in p)
    score_base = total_pts/total_max*100

    # ── Ajuste por HMM urgencia ──
    hmm_adj = {1:+8, 2:+3, 3:-8, 4:-18}.get(hmm_urgencia, 0)

    # ── Ajuste por tendencia ──
    tend_adj = 0
    tf = tendencia_info["fuerza"]
    if tf >= 80:   tend_adj = +10
    elif tf >= 65: tend_adj = +5
    elif tf <= 20: tend_adj = -15
    elif tf <= 35: tend_adj = -8

    # ── Ajuste MTF ──
    mtf_adj = 0
    if mtf_info["ok"]:
        if mtf_info["tendencia"]=="ALCISTA" and mtf_info["score_sup"]>60: mtf_adj=+8
        elif mtf_info["tendencia"]=="BAJISTA" and mtf_info["score_sup"]<40: mtf_adj=-10

    # ── Bonus de confluencia máxima (Grok) ──
    bonus = 0
    if pct<0.15 and rsi<35 and hmm_urgencia<=2 and vr>1.5:
        bonus = +15  # Confluencia perfecta de compra

    score_final = np.clip(score_base + hmm_adj + tend_adj + mtf_adj + bonus, 0, 100)
    return int(round(score_final)), p, {"hmm":hmm_adj,"tend":tend_adj,"mtf":mtf_adj,"bonus":bonus}

# ══════════════════════════════════════════════════════════════
#  ENTRADAS PRECISAS + TRAILING STOP (mejora #3 de Grok)
# ══════════════════════════════════════════════════════════════
def generar_señal_precisa(score, ind, tendencia_info, mtf_info, fuente):
    precio = ind["bb_mid"].iloc[-1]
    atr    = ind["atr"].iloc[-1]
    if pd.isna(atr) or atr==0: atr = precio*0.015
    bb_lo  = ind["bb_lo"].iloc[-1]
    bb_up  = ind["bb_up"].iloc[-1]
    is_crypto = fuente in ("binance","coingecko")
    mult   = 1.0 if is_crypto else 1.3   # cripto ATR ya es grande

    # Bloquear señales contra tendencia fuerte
    if score >= 68 and tendencia_info["bloquear_long"]:
        score = min(score, 54)  # degradar si tendencia bajista fuerte
    if score <= 32 and tendencia_info["bloquear_short"]:
        score = max(score, 40)

    if score >= 80:
        # LONG AGRESIVO — tendencia fuerte + confluencia alta
        entry  = precio
        sl     = max(precio - 1.3*mult*atr, bb_lo*0.995)
        tp1    = precio + 2.2*mult*atr
        tp2    = precio + 4.0*mult*atr
        trail  = 1.5*mult*atr
        return ("🟢 LONG AGRESIVO","sc", entry,sl,tp1,tp2,trail,
                f"Confluencia máxima. Entrada directa en {fp(entry)}. "
                f"Trailing: {fp(trail)} por vela.")

    elif score >= 68:
        # LONG CONSERVADOR — esperar pullback a zona BB inferior
        entry  = max(bb_lo*1.01, precio - 0.5*mult*atr)
        sl     = entry - 1.5*mult*atr
        tp1    = entry + 2.8*mult*atr
        tp2    = entry + 5.0*mult*atr
        trail  = 2.0*mult*atr
        return ("🟡 LONG CONSERVADOR","se", entry,sl,tp1,tp2,trail,
                f"Entrada en pullback ~{fp(entry)}. Espera retroceso antes de entrar.")

    elif score >= 52:
        return ("⚪ NEUTRAL — Observar","sn", None,None,None,None,None,
                "Sin confluencia suficiente. Mantente fuera o reduce tamaño.")

    elif score >= 35:
        # SHORT / SALIDA
        entry  = precio
        sl     = precio + 1.3*mult*atr
        tp1    = precio - 2.0*mult*atr
        tp2    = precio - 3.5*mult*atr
        trail  = 1.5*mult*atr
        return ("🔴 PRECAUCIÓN / SALIDA","sv", entry,sl,tp1,tp2,trail,
                "Debilidad confirmada. Si tienes posición, ajusta tu SL.")

    else:
        entry  = precio
        sl     = precio + 1.0*mult*atr
        tp1    = precio - 2.5*mult*atr
        tp2    = precio - 4.0*mult*atr
        trail  = 1.0*mult*atr
        return ("🚨 SALIDA URGENTE","sv", entry,sl,tp1,tp2,trail,
                "⚠️ Alta presión bajista + tendencia en contra. PROTEGE CAPITAL.")

# ══════════════════════════════════════════════════════════════
#  BACKTESTING VECTORIZADO (mejora #5 de Grok)
# ══════════════════════════════════════════════════════════════
def backtesting_simple(df, ind):
    """
    Backtest vectorizado basado en cruces MACD + RSI < 65 + EMA alineada.
    Sin lookahead bias. Retorna métricas y curva de equity.
    """
    c    = df["Close"].values
    macd = ind["macd_hist"].values
    rsi  = ind["rsi"].values
    e9   = ind["ema9"].values
    e21  = ind["ema21"].values
    atr  = ind["atr"].values
    n    = len(c)

    posicion   = 0   # 0=fuera, 1=long
    entrada    = 0.0
    sl_actual  = 0.0
    trail      = 0.0
    equity     = [1000.0]
    trades     = []
    capital    = 1000.0

    for i in range(20, n):
        if np.isnan(macd[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]): 
            equity.append(capital); continue

        atr_i  = atr[i]
        precio = c[i]

        if posicion == 0:
            # Señal de entrada: cruce MACD alcista + RSI < 65 + EMA9>EMA21
            cruce_alc = (macd[i]>0 and macd[i-1]<=0)
            ema_ok    = (e9[i]>e21[i]) if not np.isnan(e9[i]) else False
            rsi_ok    = (rsi[i]<65)
            if cruce_alc and ema_ok and rsi_ok:
                posicion  = 1
                entrada   = precio
                trail     = 1.5*atr_i
                sl_actual = entrada - trail
                trades.append({"tipo":"ENTRY","precio":precio,"idx":i})
        else:
            # Trailing stop dinámico
            nuevo_sl = precio - trail
            sl_actual = max(sl_actual, nuevo_sl)   # solo sube

            # Salida por SL o señal bajista
            cruce_baj = (macd[i]<0 and macd[i-1]>=0)
            if precio <= sl_actual or cruce_baj:
                pnl    = (precio - entrada) / entrada * 100
                capital *= (1 + pnl/100)
                trades.append({"tipo":"EXIT","precio":precio,"idx":i,"pnl_pct":pnl})
                posicion = 0

        equity.append(capital)

    # Cerrar posición abierta al final
    if posicion == 1 and len(c)>0:
        pnl = (c[-1]-entrada)/entrada*100
        capital *= (1+pnl/100)
        trades.append({"tipo":"EXIT","precio":c[-1],"idx":n-1,"pnl_pct":pnl})
        equity[-1] = capital

    # Métricas
    exits   = [t for t in trades if t["tipo"]=="EXIT"]
    n_trades= len(exits)
    if n_trades == 0:
        return {"equity":equity,"trades":trades,"n_trades":0,"win_rate":0,
                "pnl_total":0,"mejor":0,"peor":0,"profit_factor":0,"max_dd":0}

    pnls    = [t["pnl_pct"] for t in exits]
    wins    = [p for p in pnls if p>0]
    losses  = [p for p in pnls if p<=0]
    wr      = len(wins)/n_trades*100
    pnl_tot = sum(pnls)
    pf      = abs(sum(wins)/(sum(losses)+1e-9))

    # Max Drawdown
    eq_arr  = np.array(equity)
    peak    = np.maximum.accumulate(eq_arr)
    dd      = (eq_arr-peak)/(peak+1e-9)*100
    max_dd  = float(dd.min())

    return {"equity":equity,"trades":trades,"n_trades":n_trades,
            "win_rate":wr,"pnl_total":pnl_tot,"mejor":max(pnls) if pnls else 0,
            "peor":min(pnls) if pnls else 0,"profit_factor":pf,"max_dd":max_dd}

# ══════════════════════════════════════════════════════════════
#  MÓDULOS CUÁNTICOS
# ══════════════════════════════════════════════════════════════
def quantum_harmonic_oscillator(prices, n_levels=5):
    c=np.array(prices,dtype=float); N=len(c)
    mu=np.mean(c); sigma=np.std(c)+1e-10; xn=(c-mu)/sigma
    fft=np.abs(np.fft.rfft(xn-np.mean(xn))); frq=np.fft.rfftfreq(N)
    dom=frq[np.argmax(fft[1:])+1] if len(fft)>2 else 1/N
    omega=2*np.pi*max(dom,1/N)
    En=[(n+0.5)*omega for n in range(n_levels)]
    pup=[mu+np.sqrt(2*E/(omega**2+1e-10))*sigma for E in En]
    pdn=[mu-np.sqrt(2*E/(omega**2+1e-10))*sigma for E in En]
    xg=np.linspace(xn.min()-1,xn.max()+1,500)
    psi=np.exp(-xg**2)/np.sqrt(np.pi)
    for n in range(1,4):
        Hn={1:2*xg,2:4*xg**2-2,3:8*xg**3-12*xg}[n]
        psi+=np.exp(-En[n]*0.5)*Hn**2*np.exp(-xg**2)
    psi/=(psi.sum()+1e-10)
    xc=xn[-1]; Vc=0.5*omega**2*xc**2
    zona="REBOTE CUÁNTICO" if abs(xc)>1.5 else("EQUILIBRIO" if abs(xc)<0.5 else "TRANSICIÓN")
    return {"x_grid":xg,"psi_sq":psi,"price_levels_up":pup,"price_levels_down":pdn,
            "omega":omega,"mu":mu,"sigma":sigma,"x_current":xc,"V_current":Vc,"zona":zona}

def heisenberg_uncertainty(prices, ventana=20):
    c=pd.Series(prices); ventana=min(ventana,max(5,len(c)//4))
    dx=c.rolling(ventana).std()/(c.rolling(ventana).mean()+1e-10)
    ret=np.log(c/(c.shift(1)+1e-10)); dp=ret.rolling(ventana).std()*np.sqrt(ventana)
    U=dx*dp; hbar=U.median()
    Un=U/(hbar+1e-10)
    clas=pd.Series("INCERTIDUMBRE NORMAL",index=c.index)
    clas[Un<0.7]="BAJA INCERTIDUMBRE"; clas[Un>=1.3]="ALTA INCERTIDUMBRE"
    zs=(c-c.rolling(ventana).mean())/(c.rolling(ventana).std()+1e-10)
    tunel=(zs.abs()>2.0)&(Un<1.0)
    ea=clas.iloc[-1]; ua=Un.iloc[-1]
    confianza=max(0,min(100,int((1.5-min(ua,1.5))/1.5*100)))
    return {"delta_x":dx,"U_norm":Un,"hbar_mkt":hbar,"clasificacion":clas,
            "tunel":tunel,"estado_actual":ea,"confianza":confianza}

def kalman_filter(prices):
    z=np.array(prices,dtype=float); N=len(z)
    A=np.array([[1,1],[0,1]]); H=np.array([[1,0]])
    so=np.std(np.diff(z))*0.5+1e-10; sp=np.std(np.diff(z))*0.1+1e-10
    R=np.array([[so**2]]); Q=np.array([[sp**2,0],[0,(sp*0.1)**2]])
    xe=np.zeros((N,2)); xe[0]=[z[0],0]; P=np.eye(2)*so**2; Pl=[P]
    for t in range(1,N):
        xp=A@xe[t-1]; Pp=A@Pl[-1]@A.T+Q
        S=H@Pp@H.T+R; K=Pp@H.T@np.linalg.inv(S)
        xe[t]=xp+K.flatten()*(z[t]-H@xp).flatten()
        Pl.append((np.eye(2)-K@H)@Pp)
    pk=xe[:,0]; vk=xe[:,1]
    var=np.array([P[0,0] for P in Pl])
    bs=pk+2*np.sqrt(np.abs(var)); bi=pk-2*np.sqrt(np.abs(var))
    va=vk[-1]
    tend="ALCISTA ↑" if va>sp*0.5 else("BAJISTA ↓" if va<-sp*0.5 else "LATERAL →")
    dp=(z[-1]-pk[-1])/(pk[-1]+1e-10)*100
    return {"precio_kalman":pk,"velocidad":vk,"banda_sup":bs,"banda_inf":bi,
            "tendencia":tend,"diff_pct":dp,"vel_actual":va}

def quantum_entanglement(ticker_main, fuente):
    pares_ref={"BTC-USD":"₿ Bitcoin","ETH-USD":"Ξ Ethereum","SPY":"📊 S&P500",
               "GLD":"🥇 Oro","^VIX":"😰 VIX","DX-Y.NYB":"💵 USD"}
    try:
        if fuente=="binance":     df_m=binance_descargar(ticker_main,"1d",90)
        elif fuente=="coingecko": df_m=coingecko_descargar(ticker_main,90)
        else:
            t=yf.Ticker(ticker_main); df_m=t.history(period="3mo",interval="1d")
        if df_m.empty or len(df_m)<15: return None
    except: return None
    ret_main=np.log(df_m["Close"]/(df_m["Close"].shift(1)+1e-10)).dropna()
    if hasattr(ret_main.index,'tz') and ret_main.index.tz:
        ret_main.index=ret_main.index.tz_localize(None)
    ret_main.index=pd.to_datetime(ret_main.index).normalize()
    correlaciones={}
    for sym,nombre in pares_ref.items():
        if sym==ticker_main: continue
        try:
            t=yf.Ticker(sym); df2=t.history(period="3mo",interval="1d")
            if df2.empty or len(df2)<15: continue
            ret2=np.log(df2["Close"]/(df2["Close"].shift(1)+1e-10)).dropna()
            if hasattr(ret2.index,'tz') and ret2.index.tz:
                ret2.index=ret2.index.tz_localize(None)
            ret2.index=pd.to_datetime(ret2.index).normalize()
            idx=ret_main.index.intersection(ret2.index)
            if len(idx)<12: continue
            corr=np.corrcoef(ret_main.loc[idx].values,ret2.loc[idx].values)[0,1]
            if not np.isnan(corr): correlaciones[nombre]={"sym":sym,"corr":float(corr)}
        except: pass
    if len(correlaciones)<2: return None
    nombres=list(correlaciones.keys()); n=len(nombres)
    vals=[correlaciones[nm]["corr"] for nm in nombres]
    C=np.zeros((n,n))
    for i in range(n):
        for j in range(n): C[i,j]=(vals[i]*vals[j]+1)/2
    np.fill_diagonal(C,1.0)
    ev=np.linalg.eigvalsh(C); ev=ev[ev>1e-10]/ev.sum()
    Svn=-np.sum(ev*np.log(ev+1e-10)); Sn=Svn/(np.log(n)+1e-10)
    for nm in nombres:
        co=correlaciones[nm]["corr"]
        if co>0.7:     correlaciones[nm]["tipo"]=("ENTRELAZADO ↑↑","#00ff88")
        elif co>0.35:  correlaciones[nm]["tipo"]=("CORRELADO ↑","#44aaff")
        elif co>-0.35: correlaciones[nm]["tipo"]=("INDEPENDIENTE ⊥","#aaaaaa")
        elif co>-0.7:  correlaciones[nm]["tipo"]=("ANTI-CORR ↑↓","#ffaa00")
        else:          correlaciones[nm]["tipo"]=("ESPEJO CUÁNTICO","#ff3355")
    return {"correlaciones":correlaciones,"nombres":nombres,"C_matrix":C,"S_vn":Svn,"S_norm":Sn}

# ══════════════════════════════════════════════════════════════
#  UTILIDADES DE RENDERIZADO
# ══════════════════════════════════════════════════════════════
def estilizar_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors="#2a4060", labelsize=7)
    for sp in ax.spines.values(): sp.set_color("#0d1a2e")

def render_fig(fig):
    st.pyplot(fig)
    plt.close(fig)

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
st.sidebar.markdown("# 🌌 AI.LINO")
st.sidebar.markdown("**QUANTUM ENGINE v4**")
st.sidebar.caption("Filtro Tendencia · MTF · Backtest · Trailing")
st.sidebar.divider()

query = st.sidebar.text_input("🔍 Buscar:", placeholder="Monad, BTC, Tesla...", key="sq")
ticker_final=None; ticker_nombre=None; fuente=None

if query and len(query)>=2:
    with st.sidebar:
        with st.spinner("Escaneando..."):
            try: res=yf.Search(query,max_results=3,enable_fuzzy_query=True); qyf=res.quotes
            except: qyf=[]
            qbn=binance_buscar(query); qcg=coingecko_buscar(query)
        opts=[]; omap={}
        for q2 in qyf:
            sym=q2.get("symbol",""); nm=q2.get("longname") or q2.get("shortname") or sym
            lbl=f"📈 {sym} — {nm} [Yahoo]"; opts.append(lbl); omap[lbl]=(sym,nm,"yahoo")
        for p in qbn:
            lbl=f"🟡 {p['symbol']} — {p['base']}/{p['quote']} [Binance]"
            opts.append(lbl); omap[lbl]=(p["symbol"],f"{p['base']}/{p['quote']}","binance")
        for c in qcg:
            cid=c.get("id",""); nm=c.get("name",cid); sym=c.get("symbol","").upper()
            rk=c.get("market_cap_rank","?")
            lbl=f"🦎 {sym} — {nm} (#{rk}) [CoinGecko]"; opts.append(lbl)
            omap[lbl]=(cid,f"{nm}({sym})","coingecko")
        if opts:
            sel=st.radio("",opts,label_visibility="collapsed")
            if sel: ticker_final,ticker_nombre,fuente=omap[sel]; st.success(f"✅ {ticker_final}")
        else: st.warning("Sin resultados.")
else:
    favs={"MON·Monad🦎":("monad","Monad","coingecko"),
          "BTC/USDT🟡":("BTCUSDT","Bitcoin/USDT","binance"),
          "ETH/USDT🔵":("ETHUSDT","Ethereum/USDT","binance"),
          "SOL/USDT☀️":("SOLUSDT","Solana/USDT","binance"),
          "PEPE/USDT🐸":("PEPEUSDT","Pepe/USDT","binance"),
          "DOGE/USDT🐕":("DOGEUSDT","Dogecoin/USDT","binance"),
          "NVDA🟢":("NVDA","NVIDIA","yahoo"),
          "AAPL🍎":("AAPL","Apple","yahoo"),
          "TSLA⚡":("TSLA","Tesla","yahoo"),
          "SPY📊":("SPY","S&P500 ETF","yahoo")}
    fs=st.sidebar.selectbox("⭐",list(favs.keys()),label_visibility="collapsed")
    ticker_final,ticker_nombre,fuente=favs[fs]

st.sidebar.divider()
tf_key = st.sidebar.selectbox("⏱ Timeframe", list(TIMEFRAMES.keys()), index=2)
st.sidebar.markdown("**⚛️ Módulos cuánticos:**")
mod_osc = st.sidebar.toggle("1· Oscilador Armónico", value=True)
mod_hei = st.sidebar.toggle("2· Heisenberg",         value=True)
mod_kal = st.sidebar.toggle("3· Kalman",              value=True)
mod_ent = st.sidebar.toggle("4· Entrelazamiento",     value=True)
mod_bkt = st.sidebar.toggle("5· Backtesting",         value=True)
if ticker_final:
    bdg={"yahoo":"📈Yahoo","binance":"🟡Binance","coingecko":"🦎CoinGecko"}.get(fuente,"")
    st.sidebar.info(f"📌 **{ticker_final}** · {bdg}")
ejecutar = st.sidebar.button("⚛️ ANALIZAR v4", use_container_width=True)
st.sidebar.caption("⚠️ Solo educativo. No asesoría financiera.")

# ══════════════════════════════════════════════════════════════
#  EJECUCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════
if ejecutar:
    if not ticker_final: st.error("Selecciona un instrumento."); st.stop()
    _,_,dias,tf_disp = TIMEFRAMES[tf_key]
    bdg_txt = {"yahoo":"📈 Yahoo","binance":"🟡 Binance","coingecko":"🦎 CoinGecko"}.get(fuente,"")

    # ── 1. Descargar datos ────────────────────────────────────
    with st.spinner("⬇️ Descargando datos..."):
        df_raw = cargar_datos(ticker_final, fuente, tf_key)
        df_sup = cargar_datos_superior(ticker_final, fuente, tf_key)

    if df_raw is None or df_raw.empty or len(df_raw)<20:
        st.error("❌ Datos insuficientes. Prueba timeframe mayor."); st.stop()

    # ── 2. Indicadores ───────────────────────────────────────
    with st.spinner("🧮 Calculando indicadores..."):
        ind = calcular_indicadores(df_raw)

    # ── 3. Filtro de tendencia ───────────────────────────────
    tend_info = filtro_tendencia(df_raw, ind)

    # ── 4. MTF ───────────────────────────────────────────────
    with st.spinner("🔭 Análisis multi-timeframe..."):
        mtf_info = analisis_mtf(df_sup)

    # ── 5. HMM ───────────────────────────────────────────────
    with st.spinner("🤖 Entrenando HMM..."):
        try:
            best_model,states,feat_idx,lmap = entrenar_hmm(df_raw, ind)
            hmm_ok  = True
            hmm_urg = lmap[states[-1]]["urgencia"]
        except Exception as e:
            hmm_ok=False; hmm_urg=2
            st.warning(f"HMM: {e}")

    # ── 6. Score + Señal ─────────────────────────────────────
    score, desglose, ajustes = calcular_score(ind, hmm_urg, tend_info, mtf_info)
    señal, cls, entry, sl, tp1, tp2, trail, desc = generar_señal_precisa(
        score, ind, tend_info, mtf_info, fuente)

    precio_actual = df_raw["Close"].iloc[-1]
    cambio_1p     = df_raw["Close"].pct_change(1).iloc[-1]*100

    # ══════════════════════════════════════════════════════════
    #  UI — HEADER
    # ══════════════════════════════════════════════════════════
    st.markdown(f"## ⚛️ {ticker_nombre}")
    st.caption(f"{bdg_txt} · {tf_disp} · {len(df_raw)} velas · {datetime.utcnow().strftime('%H:%M UTC')}")

    pfmt = f"${precio_actual:,.6f}" if precio_actual<1 else f"${precio_actual:,.4f}" if precio_actual<10 else f"${precio_actual:,.2f}"
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("Precio", pfmt, f"{cambio_1p:+.2f}%")
    rsi_v=ind["rsi"].iloc[-1]; m2.metric("RSI(14)", f"{rsi_v:.1f}" if not pd.isna(rsi_v) else "N/D")
    av=ind["atr_pct"].iloc[-1]; m3.metric("Volatilidad", f"{av:.2f}%" if not pd.isna(av) else "N/D")
    vr=ind["vol_ratio"].iloc[-1]; m4.metric("Vol Ratio", f"{vr:.2f}x" if not pd.isna(vr) else "N/D")
    sc_col = "#00ff88" if score>=68 else("#ffaa00" if score>=50 else "#ff3355")
    m5.metric("Score v4", f"{score}/100")
    m6.metric("Tendencia", tend_info["tendencia"][:10])
    st.divider()

    # ══════════════════════════════════════════════════════════
    #  FILTRO DE TENDENCIA + MTF (nuevo — siempre visible)
    # ══════════════════════════════════════════════════════════
    ta, tb, tc = st.columns([2,1,1])
    with ta:
        st.markdown(f"""
        <div class="{tend_info['clase']}">
            <div style="font-size:0.7rem;letter-spacing:2px;opacity:0.7">FILTRO DE TENDENCIA · EMA200</div>
            <div style="font-size:1.2rem;font-weight:700;margin:4px 0">{tend_info['tendencia']}</div>
            <div style="font-size:0.78rem;opacity:0.8">
                EMA200: {fp(tend_info['ema200'])} &nbsp;|&nbsp;
                Pendiente EMA50: {tend_info['pendiente_e50']:+.2f}%
            </div>
        </div>""", unsafe_allow_html=True)
    with tb:
        fuerza_c = "#00ff88" if tend_info["fuerza"]>=65 else("#ffaa00" if tend_info["fuerza"]>=45 else "#ff3355")
        st.markdown(f'<div class="mc"><div class="ml">Fuerza Tendencia</div><div class="mv" style="color:{fuerza_c}">{tend_info["fuerza"]}/100</div><div class="bw"><div class="bf" style="width:{tend_info["fuerza"]}%;background:{fuerza_c}"></div></div></div>', unsafe_allow_html=True)
    with tc:
        if mtf_info["ok"]:
            mtf_c = "#00ff88" if mtf_info["tendencia"]=="ALCISTA" else("#ff3355" if mtf_info["tendencia"]=="BAJISTA" else "#ffaa00")
            st.markdown(f'<div class="mc"><div class="ml">MTF Superior</div><div class="mv" style="color:{mtf_c}">{mtf_info["tendencia"]}</div><div style="color:#2a4060;font-size:0.7rem">Score sup: {mtf_info["score_sup"]}/100</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="mc"><div class="ml">MTF Superior</div><div class="mv" style="color:#4a6080">N/D</div></div>', unsafe_allow_html=True)

    st.divider()

    # ══════════════════════════════════════════════════════════
    #  GRÁFICO PRINCIPAL — ancho completo
    # ══════════════════════════════════════════════════════════
    plt.style.use("dark_background")
    fig_main = plt.figure(figsize=(14,11), facecolor=BG)
    gs = gridspec.GridSpec(4,1,figure=fig_main,hspace=0.07,height_ratios=[3,1,1,1])
    ax1=fig_main.add_subplot(gs[0]); ax2=fig_main.add_subplot(gs[1],sharex=ax1)
    ax3=fig_main.add_subplot(gs[2],sharex=ax1); ax4=fig_main.add_subplot(gs[3],sharex=ax1)
    for ax in [ax1,ax2,ax3,ax4]: estilizar_ax(ax)

    idx = df_raw.index; cs = df_raw["Close"]
    if hmm_ok:
        ca = df_raw["Close"].reindex(feat_idx)
        for i in range(best_model.n_components):
            mask=states==i
            ax1.scatter(feat_idx[mask],ca[mask],color=lmap[i]["color"],
                       s=12,alpha=0.75,zorder=3,label=lmap[i]["nombre"])
    ax1.plot(idx,cs,color="#1a2a4a",lw=0.7,alpha=0.5,zorder=2)
    ax1.plot(idx,ind["ema9"],  color="#ffdd44",lw=1,  alpha=0.85,label="EMA9")
    ax1.plot(idx,ind["ema21"], color="#ff8844",lw=1,  alpha=0.85,label="EMA21")
    ax1.plot(idx,ind["ema50"], color="#cc44ff",lw=1.2,alpha=0.9, label="EMA50")
    ax1.plot(idx,ind["ema200"],color="#ffffff", lw=1.5,alpha=0.6,ls="--",label="EMA200")
    ax1.fill_between(idx,ind["bb_up"],ind["bb_lo"],alpha=0.06,color="#4488ff")
    ax1.plot(idx,ind["bb_up"],color="#224488",lw=0.7,ls="--")
    ax1.plot(idx,ind["bb_lo"],color="#224488",lw=0.7,ls="--")
    # Niveles de entrada/SL/TP
    if entry:
        ax1.axhline(entry,color="#00ff88",lw=1.2,ls="-", alpha=0.8,label=f"Entry {fp(entry)}")
        ax1.axhline(sl,   color="#ff3355",lw=1,  ls="--",alpha=0.7,label=f"SL {fp(sl)}")
        ax1.axhline(tp1,  color="#ffaa00",lw=1,  ls=":",alpha=0.7, label=f"TP1 {fp(tp1)}")
        ax1.axhline(tp2,  color="#ffdd44",lw=1,  ls=":",alpha=0.6, label=f"TP2 {fp(tp2)}")
    ax1.legend(loc="upper left",fontsize=6,framealpha=0.3,ncol=4)
    ax1.set_ylabel("Precio",color="#2a4060",fontsize=8)
    ax1.tick_params(labelbottom=False)

    cv=["#00ff88" if df_raw["Close"].iloc[i]>=df_raw["Open"].iloc[i] else "#ff3355" for i in range(len(df_raw))]
    ax2.bar(idx,df_raw["Volume"],color=cv,alpha=0.6,width=0.8)
    ax2.plot(idx,ind["vol_sma"],color="#ffaa00",lw=1,alpha=0.8)
    ax2.set_ylabel("Vol",color="#2a4060",fontsize=7); ax2.tick_params(labelbottom=False)

    ax3.plot(idx,ind["rsi"],   color="#ff8844",lw=1.2,label="RSI")
    ax3.plot(idx,ind["stoch_k"],color="#44aaff",lw=0.9,alpha=0.7,label="StochK")
    ax3.axhline(70,color="#ff3355",lw=0.7,ls="--",alpha=0.6)
    ax3.axhline(30,color="#00ff88",lw=0.7,ls="--",alpha=0.6)
    ax3.fill_between(idx,70,100,alpha=0.05,color="#ff3355")
    ax3.fill_between(idx,0,30, alpha=0.05,color="#00ff88")
    ax3.set_ylim(0,100); ax3.legend(loc="upper left",fontsize=6,framealpha=0.3)
    ax3.set_ylabel("RSI",color="#2a4060",fontsize=7); ax3.tick_params(labelbottom=False)

    ch=["#00ff88" if v>=0 else "#ff3355" for v in ind["macd_hist"]]
    ax4.bar(idx,ind["macd_hist"],color=ch,alpha=0.7,width=0.8)
    ax4.plot(idx,ind["macd"],   color="#4488ff",lw=1.2,label="MACD")
    ax4.plot(idx,ind["macd_sig"],color="#ffaa00",lw=1,  label="Signal")
    ax4.axhline(0,color="#1a2a4a",lw=0.5,ls=":")
    ax4.legend(loc="upper left",fontsize=6,framealpha=0.3)
    ax4.set_ylabel("MACD",color="#2a4060",fontsize=7)
    ax4.tick_params(labelrotation=25,labelsize=6)

    render_fig(fig_main)

    # ══════════════════════════════════════════════════════════
    #  PANEL SEÑAL + NIVELES
    # ══════════════════════════════════════════════════════════
    pa,pb,pc = st.columns([2,1,1])
    with pa:
        st.markdown(f'<div class="sbox {cls}"><div style="font-size:1.05rem;font-weight:700;margin-bottom:5px">{señal}</div><div style="opacity:0.85">{desc}</div></div>', unsafe_allow_html=True)
        if entry:
            st.markdown(f"""
            <div class="mc">
                <div class="ml">Niveles precisos (ATR × trailing)</div>
                <span style="color:#00ff88">⚡ Entry: {fp(entry)}</span>&nbsp;&nbsp;
                <span style="color:#ff3355">🛑 SL: {fp(sl)}</span>&nbsp;&nbsp;
                <span style="color:#ffaa00">🎯 TP1: {fp(tp1)}</span>&nbsp;&nbsp;
                <span style="color:#ffdd44">🎯 TP2: {fp(tp2)}</span>
                <div style="color:#2a4060;font-size:0.7rem;margin-top:4px">
                    Trailing dinámico: {fp(trail)} | Mueve SL a breakeven al llegar a TP1
                </div>
            </div>""", unsafe_allow_html=True)
    with pb:
        if hmm_ok:
            ea=lmap[states[-1]]
            st.markdown(f'<div class="mc" style="border-left:3px solid {ea["color"]}"><div class="ml">Régimen HMM</div><div class="mv" style="color:{ea["color"]};font-size:0.9rem">{ea["emoji"]} {ea["nombre"]}</div><div style="color:#2a4060;font-size:0.7rem">Perm: {best_model.transmat_[states[-1],states[-1]]*100:.0f}%</div></div>', unsafe_allow_html=True)
    with pc:
        st.markdown(f'<div class="mc"><div class="ml">Score v4</div><div class="mv" style="color:{sc_col}">{score}/100</div><div class="bw"><div class="bf" style="width:{score}%;background:{sc_col}"></div></div><div style="color:#2a4060;font-size:0.68rem">HMM:{ajustes["hmm"]:+d} Tend:{ajustes["tend"]:+d} MTF:{ajustes["mtf"]:+d} Bonus:{ajustes["bonus"]:+d}</div></div>', unsafe_allow_html=True)

    with st.expander("📊 Desglose del Score v4"):
        for nm,pts,mx,em in desglose:
            pct=pts/mx*100; col="#00ff88" if pct>=70 else("#ffaa00" if pct>=40 else "#ff3355")
            st.markdown(f'<div style="margin:3px 0"><div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#4a6080"><span>{em} {nm}</span><span style="color:{col}">{pts}/{mx}</span></div><div class="bw" style="height:6px"><div class="bf" style="width:{pct:.0f}%;background:{col}"></div></div></div>', unsafe_allow_html=True)

    st.divider()

    # ══════════════════════════════════════════════════════════
    #  BACKTESTING
    # ══════════════════════════════════════════════════════════
    if mod_bkt:
        with st.expander("📈 BACKTESTING — Estrategia MACD+RSI+EMA+Trailing", expanded=True):
            st.markdown('<div class="qcard"><div class="qtitle">Backtest Vectorizado — Sin Lookahead Bias</div>Simulación de la estrategia: <b>Entrada</b> en cruce MACD alcista + RSI &lt; 65 + EMA9 &gt; EMA21. <b>Salida</b> por trailing dinámico (1.5×ATR) o cruce MACD bajista. Capital inicial: $1,000.</div>', unsafe_allow_html=True)
            with st.spinner("Ejecutando backtest..."):
                bkt = backtesting_simple(df_raw, ind)

            if bkt["n_trades"] == 0:
                st.info("No se generaron operaciones en este período. Prueba un timeframe mayor.")
            else:
                # Gráfico equity
                fig_bkt = plt.figure(figsize=(14,5), facecolor=BG)
                gs_b = gridspec.GridSpec(1,2,figure=fig_bkt,wspace=0.1)
                ax_eq = fig_bkt.add_subplot(gs_b[0]); estilizar_ax(ax_eq)
                eq_arr = np.array(bkt["equity"])
                ax_eq.plot(eq_arr,color="#00aaff",lw=1.5,label="Equity")
                ax_eq.fill_between(range(len(eq_arr)),1000,eq_arr,
                                   where=eq_arr>=1000,alpha=0.15,color="#00ff88")
                ax_eq.fill_between(range(len(eq_arr)),1000,eq_arr,
                                   where=eq_arr<1000, alpha=0.15,color="#ff3355")
                ax_eq.axhline(1000,color="#4a6080",lw=0.8,ls="--")
                ax_eq.set_title("Curva de Equity ($1,000 inicial)",color="#4488ff",fontsize=9)
                ax_eq.legend(fontsize=7,framealpha=0.3)

                # Distribución PnL
                exits = [t for t in bkt["trades"] if t["tipo"]=="EXIT"]
                ax_pn = fig_bkt.add_subplot(gs_b[1]); estilizar_ax(ax_pn)
                if exits:
                    pnls = [t["pnl_pct"] for t in exits]
                    colors_p = ["#00ff88" if p>0 else "#ff3355" for p in pnls]
                    ax_pn.bar(range(len(pnls)),pnls,color=colors_p,alpha=0.8)
                    ax_pn.axhline(0,color="#4a6080",lw=0.8,ls=":")
                    ax_pn.set_title("PnL por Operación (%)",color="#4488ff",fontsize=9)
                plt.tight_layout(); render_fig(fig_bkt)

                # Métricas en fila
                capital_final = bkt["equity"][-1]
                ret_total = (capital_final-1000)/10
                b1,b2,b3,b4,b5,b6 = st.columns(6)
                rc = "#00ff88" if ret_total>0 else "#ff3355"
                b1.markdown(f'<div class="mc"><div class="ml">Retorno Total</div><div class="mv" style="color:{rc}">{ret_total:+.1f}%</div></div>', unsafe_allow_html=True)
                wrc = "#00ff88" if bkt["win_rate"]>50 else "#ff3355"
                b2.markdown(f'<div class="mc"><div class="ml">Win Rate</div><div class="mv" style="color:{wrc}">{bkt["win_rate"]:.0f}%</div></div>', unsafe_allow_html=True)
                b3.markdown(f'<div class="mc"><div class="ml">Operaciones</div><div class="mv">{bkt["n_trades"]}</div></div>', unsafe_allow_html=True)
                pfc = "#00ff88" if bkt["profit_factor"]>1.3 else("#ffaa00" if bkt["profit_factor"]>1 else "#ff3355")
                b4.markdown(f'<div class="mc"><div class="ml">Profit Factor</div><div class="mv" style="color:{pfc}">{bkt["profit_factor"]:.2f}</div></div>', unsafe_allow_html=True)
                b5.markdown(f'<div class="mc"><div class="ml">Mejor Op.</div><div class="mv" style="color:#00ff88">{bkt["mejor"]:+.1f}%</div></div>', unsafe_allow_html=True)
                ddc = "#ff3355" if bkt["max_dd"]<-15 else("#ffaa00" if bkt["max_dd"]<-8 else "#00ff88")
                b6.markdown(f'<div class="mc"><div class="ml">Max Drawdown</div><div class="mv" style="color:{ddc}">{bkt["max_dd"]:.1f}%</div></div>', unsafe_allow_html=True)

                # Advertencia si el backtest es muy corto
                if bkt["n_trades"] < 5:
                    st.warning("⚠️ Pocas operaciones — usa un timeframe mayor (3-6 meses) para resultados estadísticamente válidos.")

    st.divider()
    st.markdown("## ⚛️ ANÁLISIS CUÁNTICO")

    prices_arr = df_raw["Close"].values

    # ══════════════════════════════════════════════════════════
    #  MÓDULO 1 — OSCILADOR ARMÓNICO
    # ══════════════════════════════════════════════════════════
    if mod_osc:
        with st.expander("🎵 MÓDULO 1 — Oscilador Armónico · Ec. de Schrödinger", expanded=False):
            st.markdown('<div class="qcard"><div class="qtitle">Ĥψ = Eψ — Soportes/Resistencias Cuánticos</div>El precio como partícula en pozo parabólico. <b>E_n = ℏω(n+½)</b> = niveles naturales de S/R. <b>|ψ|²</b> = densidad de probabilidad del precio futuro.</div>', unsafe_allow_html=True)
            with st.spinner("Schrödinger..."):
                qho = quantum_harmonic_oscillator(prices_arr)
            fig_m1 = plt.figure(figsize=(14,5), facecolor=BG)
            ax1a=fig_m1.add_subplot(1,2,1); estilizar_ax(ax1a)
            ax1a.plot(prices_arr,color="#4488ff",lw=1.2,label="Precio")
            clv=["#00ff88","#44aaff","#ffaa00","#ff8844","#ff3355"]
            for i,(up,dn) in enumerate(zip(qho["price_levels_up"],qho["price_levels_down"])):
                if 0<up<=prices_arr.max()*1.3: ax1a.axhline(up,color=clv[i],lw=1,ls="--",alpha=0.8,label=f"E{i+1}↑{fp(up)}")
                if 0<dn>=prices_arr.min()*0.7: ax1a.axhline(dn,color=clv[i],lw=1,ls=":",alpha=0.6)
            ax1a.legend(fontsize=6,framealpha=0.3,loc="upper left")
            ax1a.set_title("Niveles de Energía — S/R Cuánticos",color="#4488ff",fontsize=9)
            ax1b=fig_m1.add_subplot(1,2,2); estilizar_ax(ax1b)
            ax1b.fill_betweenx(qho["x_grid"]*qho["sigma"]+qho["mu"],0,qho["psi_sq"],alpha=0.35,color="#4488ff")
            ax1b.plot(qho["psi_sq"],qho["x_grid"]*qho["sigma"]+qho["mu"],color="#00aaff",lw=2)
            ax1b.axhline(prices_arr[-1],color="#ffdd44",lw=1.5,ls="--",label=f"Precio:{fp(prices_arr[-1])}")
            ax1b.axhline(qho["mu"],     color="#00ff88",lw=1,  ls=":",label=f"Equil:{fp(qho['mu'])}")
            ax1b.legend(fontsize=6,framealpha=0.3)
            ax1b.set_title("|ψ|² — Probabilidad de Precio Futuro",color="#4488ff",fontsize=9)
            plt.tight_layout(); render_fig(fig_m1)
            zcol={"REBOTE CUÁNTICO":"#00ff88","EQUILIBRIO":"#4488ff","TRANSICIÓN":"#ffaa00"}.get(qho["zona"],"#aaa")
            r1,r2,r3,r4,r5,r6=st.columns(6)
            r1.markdown(f'<div class="mc" style="border-left:2px solid {zcol}"><div class="ml">Zona</div><div class="mv" style="color:{zcol};font-size:0.82rem">{qho["zona"]}</div></div>', unsafe_allow_html=True)
            r2.markdown(f'<div class="mc"><div class="ml">ω (FFT)</div><div class="mv">{qho["omega"]:.4f}</div></div>', unsafe_allow_html=True)
            r3.markdown(f'<div class="mc"><div class="ml">V(x)</div><div class="mv">{qho["V_current"]:.4f}</div></div>', unsafe_allow_html=True)
            for i,(col_r,up,dn) in enumerate(zip([r4,r5,r6],qho["price_levels_up"][:3],qho["price_levels_down"][:3])):
                col_r.markdown(f'<div class="mc"><div class="ml">E{i+1}</div><div style="color:#00ff88;font-size:0.82rem">↑{fp(up)}</div><div style="color:#ff3355;font-size:0.82rem">↓{fp(dn)}</div></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    #  MÓDULO 2 — HEISENBERG
    # ══════════════════════════════════════════════════════════
    if mod_hei:
        with st.expander("🌊 MÓDULO 2 — Principio de Incertidumbre de Heisenberg", expanded=False):
            st.markdown('<div class="qcard"><div class="qtitle">ΔP · Δx ≥ ℏ/2</div><b>U pequeño = señales confiables. U grande = caos.</b> Túnel cuántico = breakout real en zona predecible.</div>', unsafe_allow_html=True)
            with st.spinner("Heisenberg..."):
                heis = heisenberg_uncertainty(prices_arr)
            fig_m2=plt.figure(figsize=(14,7),facecolor=BG)
            gs2=gridspec.GridSpec(3,1,figure=fig_m2,hspace=0.12)
            a2a=fig_m2.add_subplot(gs2[0]); a2b=fig_m2.add_subplot(gs2[1],sharex=a2a); a2c=fig_m2.add_subplot(gs2[2],sharex=a2a)
            for axi in [a2a,a2b,a2c]: estilizar_ax(axi)
            nh=len(prices_arr)
            a2a.plot(prices_arr,color="#4488ff",lw=1.2,label="Precio")
            ti=np.where(heis["tunel"].values)[0]
            if len(ti): a2a.scatter(ti,prices_arr[ti],color="#ffdd44",s=40,zorder=5,label="⚡ Túnel")
            a2a.legend(fontsize=7,framealpha=0.3); a2a.set_title("Precio + Tunnel Cuántico",color="#4488ff",fontsize=9)
            a2b.fill_between(range(nh),heis["delta_x"].values,alpha=0.35,color="#ff8844")
            a2b.plot(heis["delta_x"].values,color="#ff8844",lw=1.2); a2b.set_title("Δx Incertidumbre",color="#ff8844",fontsize=9)
            uv=heis["U_norm"].values
            cu=["#00ff88" if not pd.isna(v) and v<0.7 else("#ffaa00" if not pd.isna(v) and v<1.3 else "#ff3355") for v in uv]
            a2c.bar(range(nh),np.nan_to_num(uv),color=cu,alpha=0.8,width=0.8)
            a2c.axhline(1.0,color="#ffaa00",lw=1,ls="--",alpha=0.7,label="ℏ")
            a2c.legend(fontsize=7,framealpha=0.3); a2c.set_title("U (🟢 predecible · 🔴 caótico)",color="#4488ff",fontsize=9)
            plt.tight_layout(); render_fig(fig_m2)
            ec={"BAJA INCERTIDUMBRE":"#00ff88","INCERTIDUMBRE NORMAL":"#ffaa00","ALTA INCERTIDUMBRE":"#ff3355"}.get(heis["estado_actual"],"#aaa")
            h1,h2,h3,h4=st.columns(4)
            h1.markdown(f'<div class="mc" style="border-left:2px solid {ec}"><div class="ml">Estado</div><div class="mv" style="color:{ec};font-size:0.8rem">{heis["estado_actual"]}</div></div>', unsafe_allow_html=True)
            h2.markdown(f'<div class="mc"><div class="ml">Confianza Operativa</div><div class="mv">{heis["confianza"]}%</div><div class="bw"><div class="bf" style="width:{heis["confianza"]}%;background:{ec}"></div></div></div>', unsafe_allow_html=True)
            h3.markdown(f'<div class="mc"><div class="ml">Túneles</div><div class="mv" style="color:#ffdd44">{int(heis["tunel"].sum())}</div></div>', unsafe_allow_html=True)
            h4.markdown(f'<div class="mc"><div class="ml">ℏ Mercado</div><div class="mv">{heis["hbar_mkt"]:.6f}</div></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    #  MÓDULO 3 — KALMAN
    # ══════════════════════════════════════════════════════════
    if mod_kal:
        with st.expander("📡 MÓDULO 3 — Filtro de Kalman", expanded=False):
            st.markdown('<div class="qcard"><div class="qtitle">Ecuaciones de Riccati — Precio Real sin Ruido</div>Estado x=[precio, velocidad]. Los cruces precio/Kalman generan señales de entrada y salida precisas.</div>', unsafe_allow_html=True)
            with st.spinner("Kalman..."):
                kal = kalman_filter(prices_arr)
            pv=prices_arr; nk=len(pv)
            dk=pv-kal["precio_kalman"]
            ck_up=np.where((dk[1:]>0)&(dk[:-1]<=0))[0]+1
            ck_dn=np.where((dk[1:]<0)&(dk[:-1]>=0))[0]+1
            fig_m3=plt.figure(figsize=(14,7),facecolor=BG)
            gs3=gridspec.GridSpec(2,1,figure=fig_m3,hspace=0.1)
            ak1=fig_m3.add_subplot(gs3[0]); ak2=fig_m3.add_subplot(gs3[1],sharex=ak1)
            for axi in [ak1,ak2]: estilizar_ax(axi)
            ak1.plot(pv,color="#1a3060",lw=1,alpha=0.7,label="Precio obs.")
            ak1.plot(kal["precio_kalman"],color="#00aaff",lw=2,label="Kalman")
            ak1.fill_between(range(nk),kal["banda_sup"],kal["banda_inf"],alpha=0.12,color="#0044ff",label="±2σ")
            ak1.plot(kal["banda_sup"],color="#224488",lw=0.7,ls="--")
            ak1.plot(kal["banda_inf"],color="#224488",lw=0.7,ls="--")
            if len(ck_up): ak1.scatter(ck_up,pv[ck_up],color="#00ff88",s=50,zorder=5,marker="^",label="Cruce ↑")
            if len(ck_dn): ak1.scatter(ck_dn,pv[ck_dn],color="#ff3355",s=50,zorder=5,marker="v",label="Cruce ↓")
            ak1.legend(fontsize=7,framealpha=0.3,loc="upper left")
            ak1.set_title("Precio Real Kalman + Cruces",color="#4488ff",fontsize=9)
            vv=kal["velocidad"]; cv_k=["#00ff88" if v>0 else "#ff3355" for v in vv]
            ak2.bar(range(nk),vv,color=cv_k,alpha=0.7,width=0.8)
            ak2.axhline(0,color="#2a4060",lw=1,ls=":")
            ak2.set_title("Velocidad d[precio]/dt",color="#4488ff",fontsize=9)
            plt.tight_layout(); render_fig(fig_m3)
            tc={"ALCISTA ↑":"#00ff88","BAJISTA ↓":"#ff3355","LATERAL →":"#ffaa00"}.get(kal["tendencia"],"#aaa")
            k1,k2,k3,k4=st.columns(4)
            k1.markdown(f'<div class="mc" style="border-left:2px solid {tc}"><div class="ml">Tendencia Kalman</div><div class="mv" style="color:{tc}">{kal["tendencia"]}</div></div>', unsafe_allow_html=True)
            dc=kal["diff_pct"]; dcc="#00ff88" if dc>0 else "#ff3355"
            k2.markdown(f'<div class="mc"><div class="ml">Precio vs Kalman</div><div class="mv" style="color:{dcc}">{dc:+.2f}%</div></div>', unsafe_allow_html=True)
            k3.markdown(f'<div class="mc"><div class="ml">Velocidad</div><div class="mv" style="color:{tc}">{kal["vel_actual"]:+.6f}</div></div>', unsafe_allow_html=True)
            k4.markdown(f'<div class="mc"><div class="ml">Cruces</div><div style="color:#00ff88;font-size:0.9rem">▲{len(ck_up)}</div><div style="color:#ff3355;font-size:0.9rem">▼{len(ck_dn)}</div></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    #  MÓDULO 4 — ENTRELAZAMIENTO
    # ══════════════════════════════════════════════════════════
    if mod_ent:
        with st.expander("🔗 MÓDULO 4 — Entrelazamiento Cuántico", expanded=False):
            st.markdown('<div class="qcard"><div class="qtitle">Entropía de Von Neumann — S = −Tr(ρ ln ρ)</div>Correlación cuántica con mercados globales. <b>Espejo cuántico (ρ≈−1)</b> = cobertura natural.</div>', unsafe_allow_html=True)
            with st.spinner("Calculando entrelazamiento..."):
                ent = quantum_entanglement(ticker_final, fuente)
            if ent is None:
                st.info("ℹ️ Sin suficientes datos diarios para correlaciones. Prueba 1D · 1 mes o mayor.")
            else:
                fig_m4=plt.figure(figsize=(14,5),facecolor=BG)
                ae1=fig_m4.add_subplot(1,2,1); estilizar_ax(ae1)
                cmap_e=LinearSegmentedColormap.from_list("q",["#ff3355",BG,"#00ff88"])
                im=ae1.imshow(ent["C_matrix"],cmap=cmap_e,vmin=0,vmax=1,aspect="auto")
                ne=ent["nombres"]
                ae1.set_xticks(range(len(ne))); ae1.set_yticks(range(len(ne)))
                ae1.set_xticklabels(ne,rotation=40,ha="right",fontsize=7,color="#4a6080")
                ae1.set_yticklabels(ne,fontsize=7,color="#4a6080")
                plt.colorbar(im,ax=ae1,shrink=0.8)
                ae1.set_title("Matriz de Densidad ρ",color="#4488ff",fontsize=9)
                ae2=fig_m4.add_subplot(1,2,2); estilizar_ax(ae2)
                corrs_e=[ent["correlaciones"][nm]["corr"] for nm in ne]
                bce=["#00ff88" if c>0.35 else("#ff3355" if c<-0.35 else "#4488ff") for c in corrs_e]
                bars_e=ae2.barh(ne,corrs_e,color=bce,alpha=0.8,height=0.55)
                ae2.axvline(0,color="#2a4060",lw=1)
                ae2.axvline(0.7,color="#00ff88",lw=0.6,ls="--",alpha=0.5)
                ae2.axvline(-0.7,color="#ff3355",lw=0.6,ls="--",alpha=0.5)
                ae2.set_xlim(-1,1); ae2.tick_params(colors="#2a4060",labelsize=8)
                ae2.set_title(f"Correlaciones con {ticker_nombre}",color="#4488ff",fontsize=9)
                for bar_e,co_e in zip(bars_e,corrs_e):
                    ae2.text(co_e+(0.03 if co_e>=0 else-0.03),bar_e.get_y()+bar_e.get_height()/2,
                             f"{co_e:.2f}",va="center",ha="left" if co_e>=0 else "right",color="white",fontsize=8)
                plt.tight_layout(); render_fig(fig_m4)
                svp=ent["S_norm"]*100
                svc="#00ff88" if svp<30 else("#ffaa00" if svp<60 else "#ff3355")
                e1,e2=st.columns([1,2])
                with e1:
                    st.markdown(f'<div class="mc"><div class="ml">Entropía Von Neumann</div><div class="mv" style="color:{svc}">{ent["S_vn"]:.3f}</div><div class="bw"><div class="bf" style="width:{svp:.0f}%;background:{svc}"></div></div><div style="color:#2a4060;font-size:0.7rem">{"Alta independencia" if svp>60 else("Entrelazado" if svp<30 else "Moderado")}</div></div>', unsafe_allow_html=True)
                with e2:
                    for nm_e,dat_e in ent["correlaciones"].items():
                        tip_e,col_e=dat_e["tipo"]
                        st.markdown(f'<div class="mc" style="padding:6px 12px;border-left:3px solid {col_e}"><div style="display:flex;justify-content:space-between;font-size:0.78rem"><span style="color:#c0d0e0">{nm_e}</span><span style="color:{col_e}">{tip_e} ρ={dat_e["corr"]:.2f}</span></div></div>', unsafe_allow_html=True)

    # ── Tabla completa ───────────────────────────────────────
    with st.expander("📋 Tabla de indicadores completa"):
        def fv(v,d=4): return f"{v:.{d}f}" if not pd.isna(v) else "N/D"
        t_data={"Indicador":["RSI(14)","MACD","MACD Hist","BB %B","ATR%","EMA9","EMA21","EMA50","EMA200","StochRSI K","Vol Ratio","Chg 1p","Chg 3p","Tend.Filtro","Tend.MTF"],
                "Valor":[fv(ind["rsi"].iloc[-1],1),fv(ind["macd"].iloc[-1],6),fv(ind["macd_hist"].iloc[-1],6),
                         fv(ind["bb_pct"].iloc[-1],3),f"{fv(ind['atr_pct'].iloc[-1],2)}%",
                         fv(ind["ema9"].iloc[-1],4),fv(ind["ema21"].iloc[-1],4),fv(ind["ema50"].iloc[-1],4),fv(ind["ema200"].iloc[-1],4),
                         fv(ind["stoch_k"].iloc[-1],1),f"{fv(ind['vol_ratio'].iloc[-1],2)}x",
                         f"{fv(ind['chg1'].iloc[-1],2)}%",f"{fv(ind['chg3'].iloc[-1],2)}%",
                         tend_info["tendencia"],mtf_info.get("tendencia","N/D")]}
        st.dataframe(pd.DataFrame(t_data),use_container_width=True,hide_index=True)


# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# 🔥 SCANNER BITSO v2 — Fuente: CoinGecko (sin bloqueos)
#    Usa /coins/markets + /coins/{id}/ohlc
#    Funciona desde Streamlit Cloud sin API key
# ══════════════════════════════════════════════════════════════

# IDs de CoinGecko para los pares disponibles en Bitso
COINS_BITSO = [
    # id_coingecko, símbolo display
    ("bitcoin",          "BTC"),
    ("ethereum",         "ETH"),
    ("ripple",           "XRP"),
    ("litecoin",         "LTC"),
    ("bitcoin-cash",     "BCH"),
    ("chainlink",        "LINK"),
    ("solana",           "SOL"),
    ("dogecoin",         "DOGE"),
    ("cardano",          "ADA"),
    ("avalanche-2",      "AVAX"),
    ("polkadot",         "DOT"),
    ("uniswap",          "UNI"),
    ("aave",             "AAVE"),
    ("shiba-inu",        "SHIB"),
    ("near",             "NEAR"),
    ("cosmos",           "ATOM"),
    ("maker",            "MKR"),
    ("compound-governance-token", "COMP"),
    ("injective-protocol","INJ"),
    ("aptos",            "APT"),
    ("arbitrum",         "ARB"),
    ("optimism",         "OP"),
    ("sui",              "SUI"),
    ("fetch-ai",         "FET"),
    ("worldcoin-wld",    "WLD"),
    ("matic-network",    "MATIC"),
    ("the-sandbox",      "SAND"),
    ("decentraland",     "MANA"),
    ("curve-dao-token",  "CRV"),
    ("synthetix-network-token","SNX"),
]

CG = "https://api.coingecko.com/api/v3"


@st.cache_data(ttl=120)   # cache 2 minutos — datos casi en tiempo real
def cg_get_markets():
    """Obtiene precios actuales, cambios y volúmenes de todos los coins."""
    ids = ",".join([c[0] for c in COINS_BITSO])
    try:
        r = requests.get(f"{CG}/coins/markets", params={
            "vs_currency": "usd",
            "ids": ids,
            "order": "market_cap_desc",
            "per_page": 50,
            "price_change_percentage": "1h,24h,7d",
            "sparkline": "false",
        }, timeout=15)
        if r.status_code == 200:
            return r.json()
        return []
    except:
        return []


@st.cache_data(ttl=300)   # cache 5 minutos
def cg_get_ohlc(coin_id, days=14):
    """Obtiene velas OHLC de CoinGecko para calcular indicadores."""
    try:
        r = requests.get(f"{CG}/coins/{coin_id}/ohlc", params={
            "vs_currency": "usd",
            "days": days,
        }, timeout=12)
        if r.status_code == 200:
            data = r.json()
            if not data or len(data) < 5:
                return None
            df = pd.DataFrame(data, columns=["ts","Open","High","Low","Close"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            df.set_index("ts", inplace=True)
            df = df.astype(float)
            df["Volume"] = 1.0   # CoinGecko OHLC no da volumen — usamos market data
            return df
        return None
    except:
        return None


def calcular_mecha_simple(precio, rsi, bb_pct, cambio_24h):
    """Mecha rápida sin OHLC — solo con datos de markets."""
    mecha_rsi = max(0, min(100, (75 - rsi) / 45 * 100)) if rsi else 50
    mecha_bb  = max(0, min(100, (1 - bb_pct) * 100))    if bb_pct is not None else 50
    # Si ya subió mucho en 24h, menos mecha
    if cambio_24h and cambio_24h > 20:
        penalizacion = 30
    elif cambio_24h and cambio_24h > 10:
        penalizacion = 15
    else:
        penalizacion = 0
    mecha = (mecha_rsi * 0.6 + mecha_bb * 0.4) - penalizacion
    return max(0, min(100, mecha))


def analizar_coin_cg(coin_data, ohlc_df):
    """
    Análisis completo usando datos de CoinGecko.
    coin_data = dict del endpoint /markets
    ohlc_df   = DataFrame de velas (puede ser None)
    """
    try:
        cid    = coin_data.get("id", "")
        sym    = coin_data.get("symbol", "").upper()
        precio = coin_data.get("current_price", 0) or 0
        cap    = coin_data.get("market_cap", 0) or 0
        vol24h = coin_data.get("total_volume", 0) or 0

        chg1h  = coin_data.get("price_change_percentage_1h_in_currency",  0) or 0
        chg24h = coin_data.get("price_change_percentage_24h",              0) or 0
        chg7d  = coin_data.get("price_change_percentage_7d_in_currency",   0) or 0

        if precio <= 0:
            return None

        # ── Cascada alcista con OHLC ──────────────────────
        cascada = 0
        rsi_v   = 50.0
        bb_pct_v= 0.5
        atr_pct = 2.0
        estabilidad = 50.0
        e9_v = e21_v = precio   # defaults

        if ohlc_df is not None and len(ohlc_df) >= 10:
            c = ohlc_df["Close"]
            o = ohlc_df["Open"]
            h = ohlc_df["High"]
            l = ohlc_df["Low"]

            # Cascada: velas verdes consecutivas
            for i in range(1, min(10, len(c))):
                if c.iloc[-i] > o.iloc[-i]:
                    cascada += 1
                else:
                    break

            # RSI
            d  = c.diff()
            g  = d.clip(lower=0).ewm(com=6, adjust=False).mean()
            ls = (-d.clip(upper=0)).ewm(com=6, adjust=False).mean()
            rsi_s = 100-(100/(1+g/(ls+1e-10)))
            rsi_v = float(rsi_s.iloc[-1]) if not pd.isna(rsi_s.iloc[-1]) else 50

            # Bollinger
            if len(c) >= 10:
                bb_m  = c.rolling(min(10,len(c))).mean()
                bb_sd = c.rolling(min(10,len(c))).std()
                bb_u  = bb_m + 2*bb_sd
                bb_l  = bb_m - 2*bb_sd
                bb_s  = (c - bb_l)/(bb_u - bb_l + 1e-9)
                bb_pct_v = float(bb_s.iloc[-1]) if not pd.isna(bb_s.iloc[-1]) else 0.5

            # ATR%
            tr  = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
            atr = tr.ewm(span=7, adjust=False).mean().iloc[-1]
            atr_pct = float(atr/precio*100) if precio>0 else 2.0

            # EMAs
            e9_v  = float(c.ewm(span=9,  adjust=False).mean().iloc[-1])
            e21_v = float(c.ewm(span=21, adjust=False).mean().iloc[-1])

            # Estabilidad: retornos consistentes
            rets = [(c.iloc[-i]-c.iloc[-i-1])/(c.iloc[-i-1]+1e-10)*100
                    for i in range(1, min(7,len(c)))]
            if rets:
                consist = sum(1 for r in rets if r>0)/len(rets)*100
                estabilidad = max(0, min(100,
                    (1-min(atr_pct,8)/8)*50 + consist*0.5))
        else:
            # Sin OHLC — estimar cascada con cambios
            if chg1h > 0 and chg24h > 0: cascada = 2
            elif chg1h > 0:              cascada = 1

        # ── Mecha restante ────────────────────────────────
        mecha = calcular_mecha_simple(precio, rsi_v, bb_pct_v, chg24h)

        # ── Score global ──────────────────────────────────
        score = 0

        # Cascada (0-28)
        if cascada >= 5:   score += 28
        elif cascada >= 3: score += 20
        elif cascada >= 2: score += 13
        elif cascada >= 1: score += 7

        # Mecha restante (0-22)
        score += int(mecha * 0.22)

        # Retornos multitf (0-20)
        if chg1h > 2:    score += 8
        elif chg1h > 0:  score += 4
        if chg24h > 8:   score += 7
        elif chg24h > 3: score += 4
        elif chg24h > 0: score += 2
        if chg7d > 15:   score += 5
        elif chg7d > 5:  score += 3

        # Volumen vs cap (proxy vol_ratio) (0-15)
        vol_ratio = vol24h / (cap + 1e-10) * 100 if cap > 0 else 0
        if vol_ratio > 15:   score += 15
        elif vol_ratio > 8:  score += 10
        elif vol_ratio > 4:  score += 6
        elif vol_ratio > 1:  score += 3

        # EMA alineación (0-10)
        if e9_v > e21_v and precio > e9_v:  score += 10
        elif e9_v > e21_v:                   score += 6
        elif precio > e21_v:                 score += 3

        # Estabilidad (0-5)
        score += int(estabilidad * 0.05)

        # Penalizaciones
        if rsi_v > 78:      score -= 15
        if bb_pct_v > 0.92: score -= 10
        if chg24h < -5:     score -= 15
        if chg1h < -2:      score -= 8
        if cascada == 0:    score -= 10

        score = max(0, min(100, score))

        # ── Estado ────────────────────────────────────────
        if score >= 70 and cascada >= 3 and mecha > 40:
            estado = "🔥 SUBIENDO FUERTE"; ec = "#00ff88"
        elif score >= 55 and cascada >= 2:
            estado = "⚡ SUBIDA ACTIVA";   ec = "#44ffaa"
        elif score >= 40 and (cascada >= 1 or chg24h > 2):
            estado = "📈 SUBIENDO";        ec = "#ffaa00"
        elif chg24h < -5 or (cascada == 0 and chg1h < -1):
            estado = "📉 BAJANDO";         ec = "#ff3355"
        else:
            estado = "↔️ LATERAL";         ec = "#4488ff"

        # ── Mecha texto ───────────────────────────────────
        if mecha >= 70:   mecha_txt = "🟢 MUCHA"
        elif mecha >= 45: mecha_txt = "🟡 MEDIA"
        elif mecha >= 25: mecha_txt = "🟠 POCA"
        else:             mecha_txt = "🔴 AGOTADA"

        # ── ATR estabilidad ───────────────────────────────
        if atr_pct < 1.5:   atr_txt = "🟢 MUY ESTABLE"
        elif atr_pct < 3:   atr_txt = "🟡 ESTABLE"
        elif atr_pct < 6:   atr_txt = "🟠 VOLÁTIL"
        else:               atr_txt = "🔴 MUY VOLÁTIL"

        return {
            "id":         cid,
            "symbol":     sym,
            "score":      score,
            "estado":     estado,
            "estado_col": ec,
            "precio":     precio,
            "chg1h":      round(chg1h,  2),
            "chg24h":     round(chg24h, 2),
            "chg7d":      round(chg7d,  2),
            "cascada":    cascada,
            "mecha":      round(mecha,  1),
            "mecha_txt":  mecha_txt,
            "rsi":        round(rsi_v,  1),
            "bb_pct":     round(bb_pct_v,3),
            "atr_pct":    round(atr_pct, 2),
            "atr_txt":    atr_txt,
            "estabilidad":round(estabilidad,1),
            "vol_ratio":  round(vol_ratio, 2),
            "e9":         e9_v,
            "e21":        e21_v,
        }
    except:
        return None


# ── UI ─────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.markdown("**🔥 SCANNER BITSO**")
with st.sidebar:
    tf_cg = st.selectbox("⏱ Período OHLC:",
        ["1 día (velas 30min)", "7 días (velas 4H)", "14 días (velas diarias)"],
        key="tf_cg", index=1)
    run_cg = st.button("🔥 ESCANEAR BITSO", use_container_width=True, key="btn_cg")

if run_cg:
    st.divider()
    st.markdown("## 🔥 SCANNER BITSO")

    dias_cg_map = {
        "1 día (velas 30min)":    1,
        "7 días (velas 4H)":      7,
        "14 días (velas diarias)":14,
    }
    dias_cg = dias_cg_map[tf_cg]

    # ── Paso 1: datos de mercado (rápido) ──────────────
    with st.spinner("📡 Obteniendo precios de mercado..."):
        market_data = cg_get_markets()
        time.sleep(1)   # respetar rate limit CoinGecko

    if not market_data:
        st.error("""
        ❌ No se pudieron obtener datos de CoinGecko.
        
        **Soluciones:**
        - Espera 60 segundos y vuelve a intentarlo (límite de peticiones)
        - CoinGecko free permite ~10-30 req/min
        """)
    else:
        # Mapear market data por id
        market_map = {c["id"]: c for c in market_data}

        # ── Paso 2: OHLC por coin ───────────────────────
        prog_cg = st.progress(0)
        resultados_cg = []
        total_cg = len(COINS_BITSO)

        for i_cg, (cid, csym) in enumerate(COINS_BITSO):
            prog_cg.progress((i_cg+1)/total_cg,
                            text=f"📊 {csym} ({i_cg+1}/{total_cg})")
            try:
                coin_mkt = market_map.get(cid)
                if not coin_mkt:
                    continue
                ohlc = cg_get_ohlc(cid, dias_cg)
                time.sleep(0.5)   # CoinGecko rate limit
                res  = analizar_coin_cg(coin_mkt, ohlc)
                if res:
                    resultados_cg.append(res)
            except:
                pass
        prog_cg.empty()

        if not resultados_cg:
            st.error("No se pudieron analizar los pares. Intenta en 1 minuto.")
        else:
            resultados_cg.sort(key=lambda x: x["score"], reverse=True)

            subiendo = [r for r in resultados_cg if "SUBIENDO" in r["estado"] or "FUERTE" in r["estado"]]
            lateral  = [r for r in resultados_cg if "LATERAL"  in r["estado"]]
            bajando  = [r for r in resultados_cg if "BAJANDO"  in r["estado"]]

            st.caption(f"✅ {len(resultados_cg)} pares analizados · 🔥 {len(subiendo)} subiendo · ↔️ {len(lateral)} lateral · 📉 {len(bajando)} bajando")

            # ════════════════════════════════════════════
            #  MAPA VISUAL DEL MERCADO
            # ════════════════════════════════════════════
            fig_mp = plt.figure(figsize=(14, 5), facecolor=BG)
            ax_mp  = fig_mp.add_subplot(1,1,1); estilizar_ax(ax_mp)

            syms_mp  = [r["symbol"] for r in resultados_cg]
            scrs_mp  = [r["score"]  for r in resultados_cg]
            cols_mp  = []
            for r in resultados_cg:
                if r["score"] >= 70:   cols_mp.append("#00ff88")
                elif r["score"] >= 55: cols_mp.append("#44ffaa")
                elif r["score"] >= 40: cols_mp.append("#ffaa00")
                elif "BAJANDO" in r["estado"]: cols_mp.append("#ff3355")
                else: cols_mp.append("#334466")

            x_mp = np.arange(len(syms_mp))
            brs_mp = ax_mp.bar(x_mp, scrs_mp, color=cols_mp, alpha=0.85, width=0.7, zorder=3)

            for xi, r in enumerate(resultados_cg):
                # Número de velas verdes
                if r["cascada"] > 0:
                    ax_mp.text(xi, r["score"]+1.5, f"🕯{r['cascada']}",
                              ha="center", va="bottom", fontsize=7, color="white")
                if r["score"] >= 50:
                    ax_mp.text(xi, r["score"]/2, str(r["score"]),
                              ha="center", va="center", fontsize=7,
                              color="black", fontweight="bold")

            ax_mp.axhline(70, color="#00ff88", lw=1,   ls="--", alpha=0.6, label="🔥 Fuerte")
            ax_mp.axhline(55, color="#ffaa00", lw=0.8, ls=":",  alpha=0.5, label="⚡ Activo")
            ax_mp.set_xticks(x_mp)
            ax_mp.set_xticklabels(syms_mp, rotation=40, ha="right", fontsize=7.5, color="#4a6080")
            ax_mp.set_ylim(0, 115)
            ax_mp.set_ylabel("Score", color="#2a4060", fontsize=8)
            ax_mp.set_title(f"Mapa del Mercado Bitso — {tf_cg} · 🕯N = velas verdes consecutivas",
                           color="#4488ff", fontsize=9)
            ax_mp.legend(fontsize=7, framealpha=0.3, loc="upper right")
            ax_mp.grid(axis="y", color="#0d1a2e", lw=0.5, alpha=0.4)
            plt.tight_layout(); render_fig(fig_mp)

            # ════════════════════════════════════════════
            #  TABS
            # ════════════════════════════════════════════
            tab_sub, tab_atr, tab_mecha, tab_todos = st.tabs([
                f"🔥 Subiendo ({len(subiendo)})",
                "📏 Más Estables (ATR)",
                "🕯 Mecha Restante",
                "📋 Todos",
            ])

            # ── TAB SUBIENDO ─────────────────────────────
            with tab_sub:
                if not subiendo:
                    st.info("No hay pares claramente alcistas ahora. Prueba '7 días (velas 4H)'.")
                else:
                    for i_row in range(0, min(9, len(subiendo)), 3):
                        fila = subiendo[i_row:i_row+3]
                        cols_f = st.columns(len(fila))
                        for col_f, r in zip(cols_f, fila):
                            pf   = r["precio"]
                            pfmt = f"${pf:,.6f}" if pf<1 else f"${pf:,.4f}" if pf<10 else f"${pf:,.2f}"
                            velas_html = "🟩"*r["cascada"] + "⬜"*max(0,6-r["cascada"])
                            c24_col = "#00ff88" if r["chg24h"] > 0 else "#ff3355"
                            c1h_col = "#00ff88" if r["chg1h"]  > 0 else "#ff3355"
                            col_f.markdown(f"""
                            <div style="background:#08090f;border:1.5px solid {r['estado_col']};
                                        border-radius:10px;padding:12px 14px;
                                        font-family:'Share Tech Mono',monospace;margin-bottom:8px">
                                <div style="font-family:'Orbitron',monospace;font-size:1rem;
                                            color:{r['estado_col']};font-weight:700">{r['symbol']}/USDT</div>
                                <div style="font-size:0.68rem;color:#4488ff">{r['estado']}</div>
                                <div style="font-size:1.3rem;font-weight:700;
                                            color:{r['estado_col']};margin:3px 0">{pfmt}</div>
                                <div style="font-size:0.9rem;margin:4px 0">
                                    {velas_html}
                                    <span style="font-size:0.68rem;color:#4a6080"> {r['cascada']} velas</span>
                                </div>
                                <div style="display:grid;grid-template-columns:1fr 1fr;
                                            gap:3px;margin-top:6px;font-size:0.72rem">
                                    <div>1H: <span style="color:{c1h_col}">{r['chg1h']:+.2f}%</span></div>
                                    <div>24H: <span style="color:{c24_col}">{r['chg24h']:+.2f}%</span></div>
                                    <div>7D: <span style="color:{'#00ff88' if r['chg7d']>0 else '#ff3355'}">{r['chg7d']:+.1f}%</span></div>
                                    <div>RSI: <span style="color:#ff8844">{r['rsi']}</span></div>
                                    <div>Mecha: <span style="color:{r['estado_col']}">{r['mecha_txt']}</span></div>
                                    <div>ATR: <span style="color:#ffaa00">{r['atr_pct']}%</span></div>
                                </div>
                                <div style="margin-top:6px;font-size:0.7rem;color:#2a4060">
                                    Score: <b style="color:{r['estado_col']};font-size:0.9rem">{r['score']}/100</b>
                                </div>
                            </div>""", unsafe_allow_html=True)

            # ── TAB ATR / ESTABILIDAD ─────────────────────
            with tab_atr:
                st.markdown("**Subidas más estables — menor ATR% = movimiento más controlado:**")
                estables = sorted(
                    [r for r in resultados_cg if r["score"]>=38 and r["chg24h"]>=0],
                    key=lambda x: x["atr_pct"]
                )
                if not estables:
                    st.info("No hay pares en subida estable ahora.")
                else:
                    # Scatter ATR% vs score
                    fig_atr = plt.figure(figsize=(14,5), facecolor=BG)
                    ax_at = fig_atr.add_subplot(1,1,1); estilizar_ax(ax_at)
                    for r in estables:
                        ax_at.scatter(r["atr_pct"], r["score"],
                                     s=r["score"]*3, color=r["estado_col"], alpha=0.8, zorder=3)
                        ax_at.annotate(r["symbol"],
                                      (r["atr_pct"], r["score"]),
                                      xytext=(4,4), textcoords="offset points",
                                      fontsize=8, color="#c0d8ff")
                    ax_at.axvline(3, color="#ffaa00", lw=1, ls="--", alpha=0.5, label="ATR 3%")
                    ax_at.axhline(60, color="#00ff88", lw=0.8, ls=":", alpha=0.5)
                    ax_at.set_xlabel("ATR% (← estable | volátil →)", color="#2a4060", fontsize=8)
                    ax_at.set_ylabel("Score", color="#2a4060", fontsize=8)
                    ax_at.set_title("Ideal: arriba-izquierda (alto score + bajo ATR)", color="#4488ff", fontsize=9)
                    ax_at.legend(fontsize=7, framealpha=0.3)
                    plt.tight_layout(); render_fig(fig_atr)

                    filas_at = []
                    for r in estables[:15]:
                        pf = r["precio"]
                        pfmt = f"${pf:,.6f}" if pf<1 else f"${pf:,.4f}" if pf<10 else f"${pf:,.2f}"
                        filas_at.append({
                            "Par":        r["symbol"]+"/USDT",
                            "ATR%":       r["atr_pct"],
                            "Estab.":     r["atr_txt"],
                            "Score":      r["score"],
                            "🕯 Velas":   r["cascada"],
                            "Mecha":      r["mecha_txt"],
                            "1H%":       f"{r['chg1h']:+.2f}%",
                            "24H%":      f"{r['chg24h']:+.2f}%",
                            "7D%":       f"{r['chg7d']:+.1f}%",
                            "RSI":        r["rsi"],
                            "Precio":     pfmt,
                        })
                    st.dataframe(pd.DataFrame(filas_at), use_container_width=True, hide_index=True,
                                column_config={
                                    "ATR%":    st.column_config.NumberColumn("ATR%", format="%.2f"),
                                    "Score":   st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
                                    "🕯 Velas":st.column_config.NumberColumn("🕯 Velas", format="%d"),
                                })

            # ── TAB MECHA ─────────────────────────────────
            with tab_mecha:
                st.markdown("**¿Cuánto le queda a cada subida?**")
                st.caption("🟢 Mucha = puede seguir. 🔴 Agotada = posible reversión pronto.")
                con_mec = sorted(
                    [r for r in resultados_cg if r["cascada"]>=1 or r["chg24h"]>2],
                    key=lambda x: x["mecha"], reverse=True
                )
                if not con_mec:
                    st.info("No hay pares en movimiento activo.")
                else:
                    fig_mec = plt.figure(figsize=(14, max(4, len(con_mec)*0.38)), facecolor=BG)
                    ax_mec  = fig_mec.add_subplot(1,1,1); estilizar_ax(ax_mec)
                    syms_mec  = [r["symbol"] for r in con_mec]
                    mechas_mec= [r["mecha"]  for r in con_mec]
                    cols_mec  = ["#00ff88" if m>=70 else("#ffaa00" if m>=45 else("#ff8844" if m>=25 else "#ff3355"))
                                 for m in mechas_mec]
                    y_pos = np.arange(len(syms_mec))
                    brs_mec = ax_mec.barh(y_pos, mechas_mec, color=cols_mec, alpha=0.85, height=0.65)
                    ax_mec.set_yticks(y_pos)
                    ax_mec.set_yticklabels(syms_mec, fontsize=8, color="#4a6080")
                    for bar, r in zip(brs_mec, con_mec):
                        ax_mec.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                                   f"{r['mecha']:.0f}% · RSI:{r['rsi']} · {r['chg24h']:+.1f}%24h",
                                   va="center", fontsize=7, color="white")
                    ax_mec.axvline(70, color="#00ff88", lw=1, ls="--", alpha=0.6, label="Mucha mecha")
                    ax_mec.axvline(25, color="#ff3355", lw=1, ls="--", alpha=0.6, label="Poca mecha")
                    ax_mec.set_xlim(0, 120)
                    ax_mec.set_title("Mecha Restante → puede seguir subiendo", color="#4488ff", fontsize=9)
                    ax_mec.legend(fontsize=7, framealpha=0.3)
                    plt.tight_layout(); render_fig(fig_mec)

                    filas_mec = []
                    for r in con_mec:
                        pf = r["precio"]
                        pfmt = f"${pf:,.6f}" if pf<1 else f"${pf:,.4f}" if pf<10 else f"${pf:,.2f}"
                        filas_mec.append({
                            "Par":       r["symbol"]+"/USDT",
                            "Mecha %":   r["mecha"],
                            "Estado":    r["mecha_txt"],
                            "RSI":       r["rsi"],
                            "BB %B":     r["bb_pct"],
                            "1H%":      f"{r['chg1h']:+.2f}%",
                            "24H%":     f"{r['chg24h']:+.2f}%",
                            "7D%":      f"{r['chg7d']:+.1f}%",
                            "Score":     r["score"],
                            "Precio":    pfmt,
                        })
                    st.dataframe(pd.DataFrame(filas_mec), use_container_width=True, hide_index=True,
                                column_config={
                                    "Mecha %": st.column_config.ProgressColumn("Mecha %", min_value=0, max_value=100, format="%.0f"),
                                    "Score":   st.column_config.ProgressColumn("Score",   min_value=0, max_value=100, format="%d"),
                                })

            # ── TAB TODOS ─────────────────────────────────
            with tab_todos:
                filas_td = []
                for r in resultados_cg:
                    pf = r["precio"]
                    pfmt = f"${pf:,.6f}" if pf<1 else f"${pf:,.4f}" if pf<10 else f"${pf:,.2f}"
                    filas_td.append({
                        "Par":      r["symbol"]+"/USDT",
                        "Estado":   r["estado"],
                        "Score":    r["score"],
                        "🕯 Velas": r["cascada"],
                        "Mecha":    r["mecha"],
                        "ATR%":     r["atr_pct"],
                        "RSI":      r["rsi"],
                        "1H%":     f"{r['chg1h']:+.2f}%",
                        "24H%":    f"{r['chg24h']:+.2f}%",
                        "7D%":     f"{r['chg7d']:+.1f}%",
                        "Precio":   pfmt,
                    })
                st.dataframe(pd.DataFrame(filas_td), use_container_width=True, hide_index=True,
                            column_config={
                                "Score":    st.column_config.ProgressColumn("Score",  min_value=0, max_value=100, format="%d"),
                                "Mecha":    st.column_config.ProgressColumn("Mecha%", min_value=0, max_value=100, format="%.0f"),
                                "🕯 Velas": st.column_config.NumberColumn("🕯 Velas", format="%d"),
                                "ATR%":     st.column_config.NumberColumn("ATR%",     format="%.2f"),
                            })

            # ── SELECTOR → ANÁLISIS + AGREGAR AL LIVE ──────
            st.divider()
            st.markdown("### 🎯 Seleccionar par")
            opciones_sel = [
                f"{r['symbol']} — Score {r['score']} · {r['estado']} · 24H: {r['chg24h']:+.1f}%"
                for r in resultados_cg
            ]
            sel_col1, sel_col2, sel_col3 = st.columns([3,1,1])
            sel_par = sel_col1.selectbox(
                "Elige un par:",
                opciones_sel, key="sel_cg",
                label_visibility="collapsed"
            )
            if sel_par:
                sym_sel = sel_par.split(" —")[0].strip()
                # Botón: agregar al monitor Live
                if sel_col2.button("📡 Agregar al Live", key="add_live_btn"):
                    st.session_state.agregar_a_live = sym_sel
                    st.success(f"✅ **{sym_sel}** agregado al monitor Live")
                # Botón: abrir en motor principal
                if sel_col3.button("⚛️ Analizar", key="analizar_btn"):
                    st.info(f"Escribe **{sym_sel}** en el buscador del sidebar → presiona ⚛️ ANALIZAR v4")



# ══════════════════════════════════════════════════════════════
# 📡 AI.LINO LIVE v2 — Monitor en Tiempo Real
#    Mejoras: fix HTML · Supabase · Scanner→Live · Auto-análisis
# ══════════════════════════════════════════════════════════════
import os

# ── Supabase (opcional — funciona sin él también) ────────────
try:
    from supabase import create_client
    SUPA_URL = os.environ.get("SUPABASE_URL", "")
    SUPA_KEY = os.environ.get("SUPABASE_KEY", "")
    if SUPA_URL and SUPA_KEY:
        supa = create_client(SUPA_URL, SUPA_KEY)
        SUPA_OK = True
    else:
        SUPA_OK = False
except:
    SUPA_OK = False

# ── Lista base de pares disponibles en Bitso ─────────────────
LIVE_PARES_BASE = [
    ("bitcoin",            "BTC"),
    ("ethereum",           "ETH"),
    ("solana",             "SOL"),
    ("ripple",             "XRP"),
    ("dogecoin",           "DOGE"),
    ("cardano",            "ADA"),
    ("avalanche-2",        "AVAX"),
    ("chainlink",          "LINK"),
    ("near",               "NEAR"),
    ("injective-protocol", "INJ"),
    ("aptos",              "APT"),
    ("arbitrum",           "ARB"),
    ("optimism",           "OP"),
    ("sui",                "SUI"),
    ("uniswap",            "UNI"),
    ("aave",               "AAVE"),
    ("fetch-ai",           "FET"),
    ("matic-network",      "MATIC"),
    ("shiba-inu",          "SHIB"),
    ("pepe",               "PEPE"),
]

# Mapa id→sym para búsqueda rápida
LIVE_ID_MAP = {cid: sym for cid, sym in LIVE_PARES_BASE}
LIVE_SYM_MAP= {sym: cid for cid, sym in LIVE_PARES_BASE}

# ── Supabase helpers ──────────────────────────────────────────
def supa_cargar_watchlist():
    """Carga la watchlist guardada en Supabase."""
    if not SUPA_OK:
        return list(LIVE_PARES_BASE)
    try:
        res = supa.table("live_watchlist").select("*").execute()
        if res.data:
            return [(r["coin_id"], r["symbol"]) for r in res.data]
    except:
        pass
    return list(LIVE_PARES_BASE)

def supa_guardar_watchlist(pares):
    """Guarda la watchlist en Supabase."""
    if not SUPA_OK:
        return False
    try:
        supa.table("live_watchlist").delete().neq("coin_id","__none__").execute()
        rows = [{"coin_id": cid, "symbol": sym} for cid, sym in pares]
        supa.table("live_watchlist").insert(rows).execute()
        return True
    except:
        return False

def supa_guardar_alerta(sym, tipo, precio, score):
    """Guarda alertas en Supabase para historial."""
    if not SUPA_OK:
        return
    try:
        supa.table("live_alertas").insert({
            "symbol":    sym,
            "tipo":      tipo,
            "precio":    float(precio),
            "score":     int(score),
            "timestamp": datetime.utcnow().isoformat(),
        }).execute()
    except:
        pass

# ── Inicializar watchlist en session_state ────────────────────
if "live_watchlist" not in st.session_state:
    st.session_state.live_watchlist = supa_cargar_watchlist()

if "live_alertas" not in st.session_state:
    st.session_state.live_alertas = []  # alertas de esta sesión


# ══════════════════════════════════════════════════════════════
#  INDICADORES PARA LIVE
# ══════════════════════════════════════════════════════════════
def calcular_adx(h, l, c, period=14):
    n = len(c)
    if n < period + 2:
        return pd.Series([np.nan]*n, index=c.index), \
               pd.Series([np.nan]*n, index=c.index), \
               pd.Series([np.nan]*n, index=c.index)
    tr   = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    dm_p = h.diff().clip(lower=0)
    dm_n = (-l.diff()).clip(lower=0)
    dm_p = dm_p.where(dm_p > dm_n, 0)
    dm_n = dm_n.where(dm_n > dm_p, 0)
    atr14= tr.ewm(span=period, adjust=False).mean()
    dip  = (dm_p.ewm(span=period,adjust=False).mean()/(atr14+1e-10))*100
    dim  = (dm_n.ewm(span=period,adjust=False).mean()/(atr14+1e-10))*100
    dx   = ((dip-dim).abs()/(dip+dim+1e-10))*100
    adx  = dx.ewm(span=period, adjust=False).mean()
    return adx, dip, dim

def pendiente_ema(ema_s, v=4):
    if len(ema_s) < v+1: return 0.0
    return round((ema_s.values[-1]-ema_s.values[-v])/(ema_s.values[-v]+1e-10)*100, 3)

def pendiente_ant(ema_s, v=4, off=2):
    if len(ema_s) < v+off+1: return 0.0
    return (ema_s.values[-1-off]-ema_s.values[-v-off])/(ema_s.values[-v-off]+1e-10)*100


@st.cache_data(ttl=55)
def live_get_markets(ids_str):
    try:
        r = requests.get(f"{CG}/coins/markets", params={
            "vs_currency":"usd", "ids":ids_str,
            "order":"market_cap_desc", "per_page":50,
            "price_change_percentage":"1h,24h", "sparkline":"false",
        }, timeout=12)
        return r.json() if r.status_code==200 else []
    except: return []

@st.cache_data(ttl=85)
def live_get_ohlc(coin_id):
    try:
        r = requests.get(f"{CG}/coins/{coin_id}/ohlc",
                        params={"vs_currency":"usd","days":7}, timeout=10)
        if r.status_code!=200 or not r.json(): return None
        df = pd.DataFrame(r.json(), columns=["ts","Open","High","Low","Close"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df.set_index("ts", inplace=True)
        return df.astype(float)
    except: return None


def analizar_live(coin_id, sym, mkt):
    try:
        precio  = mkt.get("current_price", 0) or 0
        chg1h   = mkt.get("price_change_percentage_1h_in_currency", 0) or 0
        chg24h  = mkt.get("price_change_percentage_24h", 0) or 0
        if precio <= 0: return None

        df = live_get_ohlc(coin_id)
        if df is None or len(df) < 15: return None

        c=df["Close"]; h=df["High"]; l=df["Low"]; o=df["Open"]

        e9  = c.ewm(span=9,  adjust=False).mean()
        e21 = c.ewm(span=21, adjust=False).mean()
        e50 = c.ewm(span=50, adjust=False).mean()

        p9=pendiente_ema(e9); p21=pendiente_ema(e21); p50=pendiente_ema(e50)
        pa9=pendiente_ant(e9); pa21=pendiente_ant(e21)

        d=c.diff(); g=d.clip(lower=0).ewm(com=6,adjust=False).mean()
        ls=(-d.clip(lower=0)).ewm(com=6,adjust=False).mean()
        rsi=float((100-(100/(1+g/(ls.clip(lower=1e-10))))).iloc[-1])
        rsi=round(rsi,1) if not np.isnan(rsi) else 50.0

        adx_s,dip_s,dim_s=calcular_adx(h,l,c)
        adx=round(float(adx_s.iloc[-1]),1) if not np.isnan(adx_s.iloc[-1]) else 0
        dip=round(float(dip_s.iloc[-1]),1) if not np.isnan(dip_s.iloc[-1]) else 0
        dim=round(float(dim_s.iloc[-1]),1) if not np.isnan(dim_s.iloc[-1]) else 0

        tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
        atr=tr.ewm(span=14,adjust=False).mean().iloc[-1]
        atr_pct=round(float(atr/precio*100),2) if precio>0 else 0

        cascada=0
        for i in range(1,min(10,len(c))):
            if c.iloc[-i]>o.iloc[-i]: cascada+=1
            else: break

        # Fuerza cascada 0-100
        fuerza=0
        if p9>0:  fuerza+=min(15,p9*50)
        if p21>0: fuerza+=min(15,p21*80)
        if p50>0: fuerza+=min(10,p50*100)
        if adx>35: fuerza+=25
        elif adx>25: fuerza+=18
        elif adx>15: fuerza+=10
        if dip>dim: fuerza+=min(15,(dip-dim)*0.5)
        fuerza+=min(15,cascada*3)
        if 40<=rsi<=65: fuerza+=5
        elif rsi>72: fuerza-=8
        fuerza=max(0,min(100,fuerza))

        acel9  = p9  > pa9
        acel21 = p21 > pa21
        alcista= dip>dim and adx>15 and p9>0

        # ── Score rápido para señal motor ────────────────────
        score_live=0
        if rsi<35:    score_live+=20
        elif rsi<50:  score_live+=12
        elif rsi>70:  score_live-=15
        if p9>0 and p21>0 and p50>0: score_live+=20
        elif p9>0 and p21>0:         score_live+=12
        if dip>dim and adx>20:       score_live+=20
        if cascada>=3:  score_live+=20
        elif cascada>=1:score_live+=10
        if acel9 and acel21: score_live+=10
        if chg1h>0:  score_live+=5
        if chg24h>0: score_live+=5
        score_live=max(0,min(100,score_live))

        # ── Señal de motor ───────────────────────────────────
        if score_live>=72 and fuerza>=50:
            señal_motor="🟢 MANTENER / ENTRAR"
            señal_col="#00ff88"
        elif score_live>=55 and fuerza>=35:
            señal_motor="🟡 VIGILAR"
            señal_col="#ffaa00"
        elif score_live<35 or (rsi>72 and not acel9):
            señal_motor="🔴 CONSIDERAR SALIDA"
            señal_col="#ff3355"
        elif fuerza<20 and cascada==0:
            señal_motor="🚨 SALIR — FUERZA AGOTADA"
            señal_col="#ff0000"
        else:
            señal_motor="⚪ NEUTRAL"
            señal_col="#4488ff"

        # ── Clasificación sección ────────────────────────────
        if not alcista or (cascada==0 and chg24h<-2 and p9<0):
            clasificacion="bajista"
        elif fuerza>=40 and acel9 and acel21:
            clasificacion="fuerte"
        elif fuerza>=20:
            clasificacion="perdiendo"
        else:
            clasificacion="bajista"

        if p9<-0.1 and p21<-0.05 and dip<dim:
            clasificacion="bajista"

        pfmt=f"${precio:,.6f}" if precio<1 else f"${precio:,.4f}" if precio<10 else f"${precio:,.2f}"

        return {
            "id":coin_id,"sym":sym,"precio_num":precio,"precio":pfmt,
            "chg1h":round(chg1h,2),"chg24h":round(chg24h,2),
            "cascada":cascada,"fuerza":round(fuerza,1),
            "p9":p9,"p21":p21,"p50":p50,
            "acel9":acel9,"acel21":acel21,
            "rsi":rsi,"adx":adx,"dip":dip,"dim":dim,
            "atr_pct":atr_pct,"clasificacion":clasificacion,
            "score_live":score_live,
            "señal_motor":señal_motor,"señal_col":señal_col,
        }
    except: return None


# ══════════════════════════════════════════════════════════════
#  CARD LIVE — usando st.container() en lugar de HTML puro
#  Solución definitiva al bug de HTML en columnas
# ══════════════════════════════════════════════════════════════
def render_card_live(r, col):
    """Renderiza card usando componentes nativos de Streamlit."""
    col_map = {
        "fuerte":   "#00ff88",
        "perdiendo":"#ffaa00",
        "bajista":  "#ff3355",
    }
    col_border = col_map.get(r["clasificacion"], "#4488ff")

    with col:
        # Borde visual con markdown simple
        c24_str = f"+{r['chg24h']:.2f}%" if r['chg24h']>=0 else f"{r['chg24h']:.2f}%"
        c1h_str = f"+{r['chg1h']:.2f}%"  if r['chg1h'] >=0 else f"{r['chg1h']:.2f}%"

        # Header
        st.markdown(f"""<div style="border-left:4px solid {col_border};
            padding:4px 8px;margin-bottom:4px;background:#08090f;border-radius:0 6px 6px 0">
            <span style="font-family:'Orbitron',monospace;color:{col_border};
                font-weight:700;font-size:0.95rem">{r['sym']}/USDT</span>
            <span style="float:right;color:#4a6080;font-size:0.75rem">
                {"🟩"*r['cascada']}{"⬜"*max(0,4-r['cascada'])}
            </span>
        </div>""", unsafe_allow_html=True)

        # Precio y cambio
        c24_col = "green" if r['chg24h']>=0 else "red"
        c1h_col = "green" if r['chg1h'] >=0 else "red"
        st.markdown(f"""<div style="font-size:1.1rem;font-weight:700;
            color:{col_border};padding:2px 8px">{r['precio']}
            <span style="font-size:0.78rem;color:{'#00ff88' if r['chg24h']>=0 else '#ff3355'}">
                {c24_str} 24H</span>
            <span style="font-size:0.7rem;color:{'#00ff88' if r['chg1h']>=0 else '#ff3355'}">
                · {c1h_str} 1H</span>
        </div>""", unsafe_allow_html=True)

        # Barra de fuerza
        fw = int(r['fuerza'])
        fc = "#00ff88" if fw>=60 else("#ffaa00" if fw>=35 else "#ff3355")
        st.markdown(f"""<div style="padding:0 8px">
            <div style="font-size:0.62rem;color:#4a6060;margin-bottom:2px">
                FUERZA CASCADA {fw}%</div>
            <div style="background:#0d1428;border-radius:3px;height:8px;overflow:hidden">
                <div style="width:{fw}%;height:100%;background:{fc};border-radius:3px"></div>
            </div>
        </div>""", unsafe_allow_html=True)

        # EMAs en métricas nativas — NO hay bug aquí
        ea, eb, ec2 = st.columns(3)
        def pend_fmt(v):
            return f"↑{v:+.3f}%" if v>0.02 else(f"↓{v:+.3f}%" if v<-0.02 else f"→{v:+.3f}%")
        ea.metric("EMA9",  pend_fmt(r['p9']),  delta_color="normal")
        eb.metric("EMA21", pend_fmt(r['p21']), delta_color="normal")
        ec2.metric("EMA50",pend_fmt(r['p50']), delta_color="normal")

        # ADX / DI / RSI
        d1,d2,d3,d4 = st.columns(4)
        adx_col="#00ff88" if r['adx']>25 else("#ffaa00" if r['adx']>15 else "#ff3355")
        d1.markdown(f"<div style='text-align:center;font-size:0.7rem;color:#4a6080'>ADX<br><b style='color:{adx_col}'>{r['adx']}</b></div>", unsafe_allow_html=True)
        d2.markdown(f"<div style='text-align:center;font-size:0.7rem;color:#4a6080'>+DI<br><b style='color:#00ff88'>{r['dip']}</b></div>", unsafe_allow_html=True)
        d3.markdown(f"<div style='text-align:center;font-size:0.7rem;color:#4a6080'>-DI<br><b style='color:#ff3355'>{r['dim']}</b></div>", unsafe_allow_html=True)
        d4.markdown(f"<div style='text-align:center;font-size:0.7rem;color:#4a6080'>RSI<br><b style='color:#ff8844'>{r['rsi']}</b></div>", unsafe_allow_html=True)

        # ATR + señal motor
        st.markdown(f"""<div style="padding:4px 8px;margin-top:2px;border-top:1px solid #0d1a2e">
            <span style="font-size:0.68rem;color:#4a6080">ATR día: </span>
            <span style="font-size:0.72rem;color:#ffaa00">{r['atr_pct']}%</span>
            &nbsp;&nbsp;
            <span style="font-size:0.72rem;font-weight:700;color:{r['señal_col']}">
                {r['señal_motor']}</span>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")


# ══════════════════════════════════════════════════════════════
#  SIDEBAR LIVE
# ══════════════════════════════════════════════════════════════
st.sidebar.divider()
st.sidebar.markdown("**📡 AI.LINO LIVE**")

with st.sidebar:
    intervalo_live = st.selectbox(
        "⏱ Refresh:",
        ["30 seg","60 seg","2 min","5 min","Manual"],
        index=1, key="intervalo_live"
    )
    run_live = st.button("📡 INICIAR MONITOR LIVE",
                         use_container_width=True, key="btn_live")

# ── Agregar par desde scanner al Live ─────────────────────────
if "agregar_a_live" in st.session_state and st.session_state.agregar_a_live:
    sym_add = st.session_state.agregar_a_live
    cid_add = LIVE_SYM_MAP.get(sym_add)
    if cid_add:
        wl = st.session_state.live_watchlist
        if (cid_add, sym_add) not in wl:
            wl.append((cid_add, sym_add))
            st.session_state.live_watchlist = wl
            supa_guardar_watchlist(wl)
    st.session_state.agregar_a_live = None

# ══════════════════════════════════════════════════════════════
#  MONITOR LIVE UI
# ══════════════════════════════════════════════════════════════
if run_live:
    st.divider()
    st.markdown("## 📡 AI.LINO LIVE")

    seg_map = {"30 seg":30,"60 seg":60,"2 min":120,"5 min":300,"Manual":0}
    seg = seg_map[intervalo_live]

    # ── Gestión de watchlist ──────────────────────────────────
    with st.expander("⚙️ Gestionar pares monitoreados", expanded=False):
        wl_actual = st.session_state.live_watchlist
        st.caption(f"{'✅ Conectado a Supabase' if SUPA_OK else '💾 Solo sesión (configura SUPABASE_URL y SUPABASE_KEY para persistencia)'}")

        # Mostrar pares actuales con botón eliminar
        st.markdown("**Pares en el monitor:**")
        for cid_w, sym_w in list(wl_actual):
            cw1, cw2 = st.columns([4,1])
            cw1.write(f"🔵 {sym_w}/USDT")
            if cw2.button("❌", key=f"rm_{sym_w}"):
                st.session_state.live_watchlist = [(c,s) for c,s in wl_actual if s!=sym_w]
                supa_guardar_watchlist(st.session_state.live_watchlist)
                st.rerun()

        # Agregar par manualmente
        st.markdown("**Agregar par:**")
        disponibles = [(cid,sym) for cid,sym in LIVE_PARES_BASE
                      if (cid,sym) not in st.session_state.live_watchlist]
        if disponibles:
            add_col1, add_col2 = st.columns([3,1])
            add_sel = add_col1.selectbox("",
                [f"{sym}" for _,sym in disponibles],
                label_visibility="collapsed", key="add_live_sel")
            if add_col2.button("➕ Agregar"):
                sym_new = add_sel
                cid_new = LIVE_SYM_MAP.get(sym_new)
                if cid_new:
                    wl = st.session_state.live_watchlist
                    if (cid_new,sym_new) not in wl:
                        wl.append((cid_new,sym_new))
                        st.session_state.live_watchlist = wl
                        supa_guardar_watchlist(wl)
                        st.rerun()

    # ── Cargar datos solo de la watchlist actual ──────────────
    pares_monitor = st.session_state.live_watchlist
    ids_str = ",".join([cid for cid,_ in pares_monitor])

    # Placeholder para el refresh parcial — solo los datos cambian
    data_placeholder = st.empty()

    def cargar_y_mostrar():
        with data_placeholder.container():
            with st.spinner("📡 Actualizando datos..."):
                mkts = live_get_markets(ids_str)
                if not mkts:
                    st.error("❌ Sin datos. Espera 60 seg y recarga.")
                    return

            mkt_map = {m["id"]:m for m in mkts}
            prog = st.progress(0)
            resultados_lv = []
            for i_l,(cid,sym_l) in enumerate(pares_monitor):
                prog.progress((i_l+1)/len(pares_monitor),text=f"📊 {sym_l}")
                mkt = mkt_map.get(cid)
                if mkt:
                    res = analizar_live(cid,sym_l,mkt)
                    if res: resultados_lv.append(res)
                time.sleep(0.25)
            prog.empty()

            if not resultados_lv:
                st.error("No se pudo analizar ningún par.")
                return

            fuertes   = sorted([r for r in resultados_lv if r["clasificacion"]=="fuerte"],
                               key=lambda x: x["fuerza"], reverse=True)
            perdiendo = sorted([r for r in resultados_lv if r["clasificacion"]=="perdiendo"],
                               key=lambda x: x["fuerza"], reverse=True)
            bajistas  = sorted([r for r in resultados_lv if r["clasificacion"]=="bajista"],
                               key=lambda x: x["fuerza"])

            hora = datetime.utcnow().strftime("%H:%M:%S UTC")
            st.caption(f"🕐 {hora} · ✅ {len(fuertes)} fuertes · ⚠️ {len(perdiendo)} perdiendo · 🔴 {len(bajistas)} bajistas")

            # ── Mini barra resumen ────────────────────────────
            todos_ord = sorted(resultados_lv,key=lambda x:x["fuerza"],reverse=True)
            fig_res = plt.figure(figsize=(14,2.2),facecolor=BG)
            ax_res  = fig_res.add_subplot(1,1,1); estilizar_ax(ax_res)
            cols_r  = ["#00ff88" if r["clasificacion"]=="fuerte" else
                       "#ffaa00" if r["clasificacion"]=="perdiendo" else "#ff3355"
                       for r in todos_ord]
            brs = ax_res.bar(range(len(todos_ord)),
                            [r["fuerza"] for r in todos_ord],
                            color=cols_r, alpha=0.85, width=0.7)
            for xi,(r,b) in enumerate(zip(todos_ord,brs)):
                ax_res.text(xi, b.get_height()+1, str(int(r["fuerza"])),
                           ha="center",fontsize=6,color="white")
            ax_res.set_xticks(range(len(todos_ord)))
            ax_res.set_xticklabels([r["sym"] for r in todos_ord],
                                   rotation=35,ha="right",fontsize=7,color="#4a6080")
            ax_res.set_ylim(0,115); ax_res.set_yticks([])
            ax_res.axhline(40,color="#ffaa00",lw=0.8,ls="--",alpha=0.5)
            ax_res.set_title("Fuerza de Cascada — Tiempo Real",
                            color="#4488ff",fontsize=9)
            plt.tight_layout(); render_fig(fig_res)

            # ── SECCIÓN 1: CASCADA FUERTE ─────────────────────
            if fuertes:
                st.markdown(f"""<div style="background:#001a0a;border:1px solid #00ff88;
                    border-radius:8px;padding:8px 14px;margin:8px 0;
                    font-family:'Orbitron',monospace;font-size:0.88rem">
                    ✅ CASCADA FUERTE — mantener o entrar
                    <span style="float:right;color:#00ff88">{len(fuertes)} pares</span>
                </div>""", unsafe_allow_html=True)

                for i in range(0, len(fuertes), 3):
                    row = fuertes[i:i+3]
                    cols_row = st.columns(len(row))
                    for r,col in zip(row, cols_row):
                        render_card_live(r, col)

            # ── SECCIÓN 2: PERDIENDO FUERZA ───────────────────
            if perdiendo:
                st.markdown(f"""<div style="background:#1a1200;border:1px solid #ffaa00;
                    border-radius:8px;padding:8px 14px;margin:8px 0;
                    font-family:'Orbitron',monospace;font-size:0.88rem">
                    ⚠️ PERDIENDO FUERZA — vigilar para salir
                    <span style="float:right;color:#ffaa00">{len(perdiendo)} pares</span>
                </div>""", unsafe_allow_html=True)

                for i in range(0, len(perdiendo), 3):
                    row = perdiendo[i:i+3]
                    cols_row = st.columns(len(row))
                    for r,col in zip(row, cols_row):
                        render_card_live(r, col)

            # ── SECCIÓN 3: BAJISTA ────────────────────────────
            if bajistas:
                st.markdown(f"""<div style="background:#1a0005;border:1px solid #ff3355;
                    border-radius:8px;padding:8px 14px;margin:8px 0;
                    font-family:'Orbitron',monospace;font-size:0.88rem">
                    🔴 TENDENCIA BAJISTA — considerar salida
                    <span style="float:right;color:#ff3355">{len(bajistas)} pares</span>
                </div>""", unsafe_allow_html=True)

                for i in range(0, len(bajistas), 3):
                    row = bajistas[i:i+3]
                    cols_row = st.columns(len(row))
                    for r,col in zip(row, cols_row):
                        render_card_live(r, col)

            # ── Alertas automáticas ───────────────────────────
            alertas_nuevas = []
            for r in resultados_lv:
                if "SALIR" in r["señal_motor"] or "AGOTADA" in r["señal_motor"]:
                    alertas_nuevas.append(r)
                    supa_guardar_alerta(r["sym"],r["señal_motor"],r["precio_num"],r["score_live"])

            if alertas_nuevas:
                st.markdown("### 🚨 ALERTAS DE SALIDA")
                for r in alertas_nuevas:
                    st.error(f"🚨 **{r['sym']}** — {r['señal_motor']} · Precio: {r['precio']} · RSI: {r['rsi']} · Fuerza: {r['fuerza']}%")

            # ── Tabla resumen ─────────────────────────────────
            with st.expander("📋 Tabla resumen"):
                filas_lv=[]
                for r in todos_ord:
                    ic={"fuerte":"✅","perdiendo":"⚠️","bajista":"🔴"}.get(r["clasificacion"],"↔️")
                    filas_lv.append({
                        "Par":r["sym"]+"/USDT","Estado":ic,
                        "Fuerza":r["fuerza"],"🕯":r["cascada"],
                        "Motor":r["señal_motor"],
                        "1H%":f"{r['chg1h']:+.2f}%","24H%":f"{r['chg24h']:+.2f}%",
                        "EMA9":r["p9"],"EMA21":r["p21"],
                        "ADX":r["adx"],"+DI":r["dip"],"-DI":r["dim"],
                        "RSI":r["rsi"],"ATR%":r["atr_pct"],
                    })
                st.dataframe(pd.DataFrame(filas_lv),use_container_width=True,hide_index=True,
                    column_config={
                        "Fuerza":st.column_config.ProgressColumn("Fuerza%",min_value=0,max_value=100,format="%.0f"),
                        "🕯":st.column_config.NumberColumn("🕯 Velas",format="%d"),
                        "EMA9":st.column_config.NumberColumn("EMA9%",format="%.3f"),
                        "EMA21":st.column_config.NumberColumn("EMA21%",format="%.3f"),
                    })

    # Primera carga
    cargar_y_mostrar()

    # ── Refresh parcial en la parte inferior ──────────────────
    if seg > 0:
        st.markdown("---")
        refresh_col1, refresh_col2 = st.columns([3,1])
        refresh_col1.info(f"⏱ Auto-refresh cada {intervalo_live} · Solo se actualizan los datos, no la página completa.")
        if refresh_col2.button("🔄 Actualizar ahora", key="refresh_manual"):
            live_get_markets.clear()
            live_get_ohlc.clear()
            st.rerun()
        time.sleep(seg)
        # Limpiar cache de datos para forzar nuevos datos
        live_get_markets.clear()
        st.rerun()


# ╔══════════════════════════════════════════════════════════════╗
# ║   AI.LINO APEX — SISTEMA DE ENTRADA DE ALTA PRECISIÓN      ║
# ║   Alto riesgo · Alto rendimiento · 3 filtros en cascada    ║
# ║   Trailing dinámico · Señal de salida automática           ║
# ╚══════════════════════════════════════════════════════════════╝
#
#  FILOSOFÍA:
#  No entrar hasta que los 3 filtros pasen simultáneamente.
#  Una vez dentro, el sistema gestiona solo hasta la salida.
#
#  FILTRO 1 — TENDENCIA BASE    : EMA200 + ADX > 25
#  FILTRO 2 — IMPULSO ACTIVO    : MACD cruce + Kalman acelerando + Vol 2x
#  FILTRO 3 — ENTRADA PRECISA   : RSI 32-58 + BB bajo + Ruptura EMA21
#
#  SALIDA AUTOMÁTICA por cualquiera de estos:
#  · Mecha agotada  : RSI > 74 + BB > 0.88 + ADX cae
#  · Tendencia rota : EMA9 cruza EMA21 hacia abajo
#  · SL dinámico    : precio toca trailing stop (ATR × 1.5)
#  · Divergencia    : precio sube pero MACD y Kalman bajan (techo)

# ── Supabase para guardar señales APEX ──────────────────────
def supa_guardar_apex_signal(sym, tipo, precio, score, sl, tp1, tp2, detalles):
    """Guarda señal APEX en Supabase para historial y seguimiento."""
    if not SUPA_OK:
        return
    try:
        supa.table("ailino_apex_signals").insert({
            "symbol":    sym,
            "tipo":      tipo,
            "precio":    float(precio),
            "score":     int(score),
            "sl":        float(sl) if sl else 0,
            "tp1":       float(tp1) if tp1 else 0,
            "tp2":       float(tp2) if tp2 else 0,
            "detalles":  str(detalles),
            "timestamp": datetime.utcnow().isoformat(),
        }).execute()
    except:
        pass

# Inicializar historial APEX en session_state
if "apex_historial" not in st.session_state:
    st.session_state.apex_historial = []
if "apex_posiciones" not in st.session_state:
    st.session_state.apex_posiciones = {}  # sym → {entry, sl, tp1, tp2, trail}


# ══════════════════════════════════════════════════════════════
#  MOTOR APEX — análisis de alta precisión por par
# ══════════════════════════════════════════════════════════════
def apex_analizar(df, sym):
    """
    Los 3 filtros en cascada + señal de salida.
    Retorna dict completo con todo lo necesario para operar.
    """
    if df is None or df.empty or len(df) < 30:
        return None
    try:
        c = df["Close"]; h = df["High"]; l = df["Low"]; o = df["Open"]
        v = df["Volume"]; n = len(c)

        # ════════════════════════════════════════════════════
        #  INDICADORES BASE
        # ════════════════════════════════════════════════════
        # EMAs
        e9   = c.ewm(span=9,   adjust=False).mean()
        e21  = c.ewm(span=21,  adjust=False).mean()
        e50  = c.ewm(span=50,  adjust=False).mean()
        e200 = c.ewm(span=200, adjust=False).mean()

        # RSI con EWM (más reactivo)
        d  = c.diff()
        g  = d.clip(lower=0).ewm(com=13, adjust=False).mean()
        ls = (-d.clip(upper=0)).ewm(com=13, adjust=False).mean()
        rsi = 100 - (100/(1 + g/(ls+1e-10)))

        # MACD
        ef   = c.ewm(span=12, adjust=False).mean()
        es   = c.ewm(span=26, adjust=False).mean()
        macd = ef - es
        msig = macd.ewm(span=9, adjust=False).mean()
        mhist= macd - msig

        # Bollinger
        bb_m = c.rolling(20).mean()
        bb_s = c.rolling(20).std()
        bb_u = bb_m + 2*bb_s
        bb_l = bb_m - 2*bb_s
        bb_p = (c - bb_l)/(bb_u - bb_l + 1e-9)

        # ATR
        tr  = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
        atr = tr.ewm(span=14, adjust=False).mean()

        # ADX + DI
        dm_p = h.diff().clip(lower=0)
        dm_n = (-l.diff()).clip(lower=0)
        dm_p = dm_p.where(dm_p > dm_n, 0)
        dm_n = dm_n.where(dm_n > dm_p, 0)
        atr14= tr.ewm(span=14, adjust=False).mean()
        dip  = (dm_p.ewm(span=14,adjust=False).mean()/(atr14+1e-10))*100
        dim  = (dm_n.ewm(span=14,adjust=False).mean()/(atr14+1e-10))*100
        dx   = ((dip-dim).abs()/(dip+dim+1e-10))*100
        adx  = dx.ewm(span=14, adjust=False).mean()

        # Kalman (velocidad y aceleración)
        pk   = c.ewm(span=5,  adjust=False).mean()
        vel  = pk.diff()
        acc  = vel.diff()

        # Stoch RSI
        rsi_mn = rsi.rolling(14).min()
        rsi_mx = rsi.rolling(14).max()
        stoch  = (rsi - rsi_mn)/(rsi_mx - rsi_mn + 1e-9)*100
        stoch_k= stoch.rolling(3).mean()

        # Volumen
        vol_sma  = v.rolling(20).mean()
        vol_ratio= v/(vol_sma+1e-10)

        # OBV (On Balance Volume)
        obv_delta = np.where(c.diff()>0, v, np.where(c.diff()<0, -v, 0))
        obv       = pd.Series(obv_delta, index=c.index).cumsum()
        obv_trend = obv.diff(3)

        # Valores actuales
        cv  = c.iloc[-1];  c2  = c.iloc[-2]
        e9v = e9.iloc[-1]; e21v= e21.iloc[-1]
        e50v= e50.iloc[-1];e200v=e200.iloc[-1]
        rsi_v   = float(rsi.iloc[-1])   if not np.isnan(rsi.iloc[-1])   else 50
        rsi_p   = float(rsi.iloc[-2])   if not np.isnan(rsi.iloc[-2])   else 50
        mh_v    = float(mhist.iloc[-1]) if not np.isnan(mhist.iloc[-1]) else 0
        mh_p    = float(mhist.iloc[-2]) if not np.isnan(mhist.iloc[-2]) else 0
        mh_p2   = float(mhist.iloc[-3]) if n>3 and not np.isnan(mhist.iloc[-3]) else mh_p
        bb_v    = float(bb_p.iloc[-1])  if not np.isnan(bb_p.iloc[-1])  else 0.5
        adx_v   = float(adx.iloc[-1])   if not np.isnan(adx.iloc[-1])   else 0
        adx_p   = float(adx.iloc[-2])   if not np.isnan(adx.iloc[-2])   else 0
        dip_v   = float(dip.iloc[-1])   if not np.isnan(dip.iloc[-1])   else 0
        dim_v   = float(dim.iloc[-1])   if not np.isnan(dim.iloc[-1])   else 0
        atr_v   = float(atr.iloc[-1])   if not np.isnan(atr.iloc[-1])   else cv*0.02
        vel_v   = float(vel.iloc[-1])   if not np.isnan(vel.iloc[-1])   else 0
        vel_p   = float(vel.iloc[-2])   if not np.isnan(vel.iloc[-2])   else 0
        acc_v   = float(acc.iloc[-1])   if not np.isnan(acc.iloc[-1])   else 0
        vr_v    = float(vol_ratio.iloc[-1]) if not np.isnan(vol_ratio.iloc[-1]) else 1
        sk_v    = float(stoch_k.iloc[-1])   if not np.isnan(stoch_k.iloc[-1])   else 50
        obv_t   = float(obv_trend.iloc[-1]) if not np.isnan(obv_trend.iloc[-1]) else 0

        # Pendientes EMA
        p9  = (e9.iloc[-1]  - e9.iloc[-4])  /(e9.iloc[-4]+1e-10)*100  if n>4 else 0
        p21 = (e21.iloc[-1] - e21.iloc[-4]) /(e21.iloc[-4]+1e-10)*100 if n>4 else 0
        p50 = (e50.iloc[-1] - e50.iloc[-4]) /(e50.iloc[-4]+1e-10)*100 if n>4 else 0

        # Cascada velas verdes
        cascada = 0
        for i in range(1, min(10,n)):
            if c.iloc[-i] > o.iloc[-i]: cascada+=1
            else: break

        atr_pct = atr_v/cv*100

        # ════════════════════════════════════════════════════
        #  FILTRO 1 — TENDENCIA BASE
        #  Necesita: precio > EMA200 + EMAs alineadas + ADX fuerte
        # ════════════════════════════════════════════════════
        f1_score = 0
        f1_det   = {}

        # Precio sobre EMA200
        sobre_e200 = cv > e200v
        if sobre_e200:
            gap200 = (cv-e200v)/(e200v+1e-10)*100
            if gap200 > 5:   f1_score+=25; f1_det["EMA200"]="🟢 +{:.1f}% sobre".format(gap200)
            elif gap200 > 1: f1_score+=18; f1_det["EMA200"]="🟡 +{:.1f}% sobre".format(gap200)
            else:            f1_score+=10; f1_det["EMA200"]="🟡 Recién cruzó"
        else:
            f1_det["EMA200"]="🔴 Bajo EMA200"

        # Alineación EMAs
        if e9v > e21v > e50v > e200v:
            f1_score+=25; f1_det["EMAs"]="🟢 Alineación perfecta ↑"
        elif e9v > e21v > e50v:
            f1_score+=18; f1_det["EMAs"]="🟡 EMA9>21>50 ↑"
        elif e9v > e21v:
            f1_score+=10; f1_det["EMAs"]="🟡 EMA9>21"
        else:
            f1_det["EMAs"]="🔴 Desalineadas"

        # ADX tendencia fuerte
        if adx_v > 35:   f1_score+=25; f1_det["ADX"]="🟢 {:.0f} MUY FUERTE".format(adx_v)
        elif adx_v > 25: f1_score+=18; f1_det["ADX"]="🟡 {:.0f} Fuerte".format(adx_v)
        elif adx_v > 18: f1_score+=8;  f1_det["ADX"]="🟡 {:.0f} Moderado".format(adx_v)
        else:            f1_det["ADX"]="🔴 {:.0f} Débil".format(adx_v)

        # DI+ vs DI-
        if dip_v > dim_v + 5:  f1_score+=15; f1_det["DI"]="🟢 +DI {:.0f} domina".format(dip_v)
        elif dip_v > dim_v:    f1_score+=8;  f1_det["DI"]="🟡 +DI leve"
        else:                  f1_det["DI"]="🔴 -DI domina"

        # Pendiente EMA50
        if p50 > 0.3:   f1_score+=10; f1_det["Pendiente"]="🟢 EMA50 sube {:.3f}%".format(p50)
        elif p50 > 0:   f1_score+=5;  f1_det["Pendiente"]="🟡 EMA50 plana"
        else:           f1_det["Pendiente"]="🔴 EMA50 baja"

        f1_max   = 100
        f1_pct   = min(100, f1_score)
        f1_pasa  = f1_pct >= 55

        # ════════════════════════════════════════════════════
        #  FILTRO 2 — IMPULSO ACTIVO
        #  Necesita: MACD + Kalman + Volumen + OBV + Cascada
        # ════════════════════════════════════════════════════
        f2_score = 0
        f2_det   = {}

        # MACD cruce o aceleración
        cruce_macd = mh_p<=0 and mh_v>0
        acel_macd  = mh_v>0 and mh_v>mh_p>mh_p2
        if cruce_macd:
            f2_score+=30; f2_det["MACD"]="🟢 CRUCE ALCISTA ⚡"
        elif acel_macd:
            f2_score+=22; f2_det["MACD"]="🟢 Acelerando ↑↑"
        elif mh_v > 0 and mh_v > mh_p:
            f2_score+=14; f2_det["MACD"]="🟡 Positivo y creciendo"
        elif mh_v > 0:
            f2_score+=7;  f2_det["MACD"]="🟡 Positivo"
        else:
            f2_det["MACD"]="🔴 Negativo"

        # Kalman velocidad y aceleración
        if vel_v > 0 and acc_v > 0:
            f2_score+=25; f2_det["Kalman"]="🟢 Vel+ · Acel+ ↑↑"
        elif vel_v > 0 and vel_v > vel_p:
            f2_score+=18; f2_det["Kalman"]="🟢 Acelerando ↑"
        elif vel_v > 0:
            f2_score+=10; f2_det["Kalman"]="🟡 Velocidad positiva"
        else:
            f2_det["Kalman"]="🔴 Decelerando"

        # Volumen spike
        if vr_v > 2.5:  f2_score+=25; f2_det["Volumen"]="🟢 SPIKE {:.1f}x ⚡".format(vr_v)
        elif vr_v > 1.8: f2_score+=18; f2_det["Volumen"]="🟢 Alto {:.1f}x".format(vr_v)
        elif vr_v > 1.3: f2_score+=10; f2_det["Volumen"]="🟡 Elevado {:.1f}x".format(vr_v)
        elif vr_v > 0.8: f2_score+=4;  f2_det["Volumen"]="⚪ Normal {:.1f}x".format(vr_v)
        else:            f2_det["Volumen"]="🔴 Seco {:.1f}x".format(vr_v)

        # OBV acumulación
        if obv_t > 0:   f2_score+=12; f2_det["OBV"]="🟢 Acumulación neta"
        else:           f2_det["OBV"]="🔴 Distribución"

        # Cascada velas
        if cascada >= 4: f2_score+=8;  f2_det["Cascada"]="🟢 {} velas ↑".format(cascada)
        elif cascada>=2: f2_score+=5;  f2_det["Cascada"]="🟡 {} velas ↑".format(cascada)
        else:            f2_det["Cascada"]="⚪ {} velas".format(cascada)

        f2_pct  = min(100, f2_score)
        f2_pasa = f2_pct >= 55

        # ════════════════════════════════════════════════════
        #  FILTRO 3 — ENTRADA PRECISA
        #  Necesita: RSI en zona correcta + BB bajo + EMA21
        # ════════════════════════════════════════════════════
        f3_score = 0
        f3_det   = {}

        # RSI zona ideal (saliendo de sobreventa, no sobrecomprado)
        if 28 <= rsi_v <= 48 and rsi_v > rsi_p:
            f3_score+=30; f3_det["RSI"]="🟢 {:.0f} — Saliendo sobreventa ↑".format(rsi_v)
        elif 28 <= rsi_v <= 55:
            f3_score+=20; f3_det["RSI"]="🟡 {:.0f} — Zona sana".format(rsi_v)
        elif rsi_v < 28:
            f3_score+=15; f3_det["RSI"]="🟡 {:.0f} — Sobreventa extrema".format(rsi_v)
        elif rsi_v > 70:
            f3_det["RSI"]="🔴 {:.0f} — Sobrecomprado".format(rsi_v)
        else:
            f3_score+=8; f3_det["RSI"]="⚪ {:.0f} — Zona alta".format(rsi_v)

        # Posición en Bollinger
        if bb_v < 0.15:
            f3_score+=25; f3_det["BB"]="🟢 {:.2f} — Bajo banda inf".format(bb_v)
        elif bb_v < 0.35:
            f3_score+=18; f3_det["BB"]="🟢 {:.2f} — Zona baja".format(bb_v)
        elif bb_v < 0.55:
            f3_score+=10; f3_det["BB"]="🟡 {:.2f} — Centro".format(bb_v)
        elif bb_v > 0.85:
            f3_det["BB"]="🔴 {:.2f} — Sobre banda sup".format(bb_v)
        else:
            f3_score+=4; f3_det["BB"]="🟡 {:.2f} — Zona alta".format(bb_v)

        # Ruptura EMA21 (precio acaba de cruzar desde abajo)
        cruce_e21 = c2 <= e21.iloc[-2] and cv > e21v
        sobre_e21 = cv > e21v
        gap21     = (cv-e21v)/(e21v+1e-10)*100
        if cruce_e21:
            f3_score+=30; f3_det["EMA21"]="🟢 RUPTURA EMA21 ⚡"
        elif sobre_e21 and gap21 < 2:
            f3_score+=18; f3_det["EMA21"]="🟢 Recién sobre EMA21"
        elif sobre_e21 and gap21 < 5:
            f3_score+=10; f3_det["EMA21"]="🟡 Sobre EMA21 +{:.1f}%".format(gap21)
        else:
            f3_det["EMA21"]="🔴 Bajo EMA21"

        # StochRSI zona entrada
        if sk_v < 20 and sk_v > stoch_k.iloc[-2] if not np.isnan(stoch_k.iloc[-2]) else False:
            f3_score+=15; f3_det["StochRSI"]="🟢 {:.0f} — Cruce alcista".format(sk_v)
        elif sk_v < 30:
            f3_score+=8;  f3_det["StochRSI"]="🟡 {:.0f} — Sobreventa".format(sk_v)
        elif sk_v > 80:
            f3_det["StochRSI"]="🔴 {:.0f} — Sobrecompra".format(sk_v)
        else:
            f3_score+=4; f3_det["StochRSI"]="⚪ {:.0f} — Neutral".format(sk_v)

        f3_pct  = min(100, f3_score)
        f3_pasa = f3_pct >= 55

        # ════════════════════════════════════════════════════
        #  SCORE APEX GLOBAL (0-100)
        # ════════════════════════════════════════════════════
        # Ponderación: F1×30% + F2×35% + F3×35%
        apex_score = int(f1_pct*0.30 + f2_pct*0.35 + f3_pct*0.35)

        # Bonus de confluencia máxima — todos los filtros fuertes
        if f1_pct>=70 and f2_pct>=70 and f3_pct>=70:
            apex_score = min(100, apex_score + 12)

        # Penalización si algún filtro falla completamente
        if not f1_pasa: apex_score = min(apex_score, 45)
        if not f2_pasa: apex_score = min(apex_score, 50)
        if not f3_pasa: apex_score = min(apex_score, 48)

        todos_pasan = f1_pasa and f2_pasa and f3_pasa

        # ════════════════════════════════════════════════════
        #  NIVELES DE ENTRADA (ATR-based, alta precisión)
        # ════════════════════════════════════════════════════
        entry = cv
        # SL dinámico: máximo entre EMA21-buffer y precio-1.5ATR
        sl    = max(e21v * 0.992, cv - 1.5*atr_v)
        # TPs progresivos para maximizar ganancia en tendencias fuertes
        tp1   = cv + 2.0*atr_v    # objetivo conservador
        tp2   = cv + 4.0*atr_v    # objetivo moderado
        tp3   = cv + 7.0*atr_v    # objetivo agresivo (dejar correr)
        trail = 1.5*atr_v          # trailing stop dinámico

        rr1   = (tp1-entry)/(entry-sl+1e-10)
        rr2   = (tp2-entry)/(entry-sl+1e-10)

        # ════════════════════════════════════════════════════
        #  SEÑALES DE SALIDA AUTOMÁTICA
        # ════════════════════════════════════════════════════
        salidas = []

        # 1. Mecha agotada
        if rsi_v > 74 and bb_v > 0.88 and adx_v < adx_p:
            salidas.append(("🔴 MECHA AGOTADA",
                           "RSI:{:.0f} · BB:{:.2f} · ADX cayendo".format(rsi_v,bb_v)))

        # 2. Tendencia rota — cruce bajista EMA9/EMA21
        if e9.iloc[-2] >= e21.iloc[-2] and e9v < e21v:
            salidas.append(("🔴 TENDENCIA ROTA",
                           "EMA9 cruzó EMA21 hacia abajo ↓"))

        # 3. Divergencia bajista (precio sube pero indicadores bajan)
        if len(c) > 5:
            precio_sube = cv > c.iloc[-5]
            macd_baja   = mhist.iloc[-1] < mhist.iloc[-5]
            kalman_baja = vel_v < 0
            if precio_sube and macd_baja and kalman_baja:
                salidas.append(("⚠️ DIVERGENCIA BAJISTA",
                               "Precio ↑ pero MACD+Kalman ↓ — posible techo"))

        # 4. MACD cruce bajista
        if mhist.iloc[-2] >= 0 and mhist.iloc[-1] < 0:
            salidas.append(("🔴 MACD CRUCE BAJISTA",
                           "Histograma cruzó a negativo"))

        # ════════════════════════════════════════════════════
        #  CLASIFICACIÓN FINAL
        # ════════════════════════════════════════════════════
        if todos_pasan and apex_score >= 80:
            clasificacion = "🚀 ENTRADA ÓPTIMA"
            col_apex      = "#00ff88"
            accion        = "ENTRAR AHORA"
        elif todos_pasan and apex_score >= 65:
            clasificacion = "⚡ ENTRADA BUENA"
            col_apex      = "#44ffaa"
            accion        = "ENTRAR CON TAMAÑO REDUCIDO"
        elif f1_pasa and f2_pasa and apex_score >= 50:
            clasificacion = "👀 CASI LISTA"
            col_apex      = "#ffaa00"
            accion        = "ESPERAR F3"
        elif f1_pasa and apex_score >= 35:
            clasificacion = "⏳ EN FORMACIÓN"
            col_apex      = "#4488ff"
            accion        = "MONITOREAR"
        else:
            clasificacion = "❌ NO CUMPLE"
            col_apex      = "#ff3355"
            accion        = "NO ENTRAR"

        pfmt = (f"${cv:,.6f}" if cv<1 else
                f"${cv:,.4f}" if cv<10 else
                f"${cv:,.2f}")

        return {
            "sym":          sym,
            "precio":       cv,
            "precio_fmt":   pfmt,
            "apex_score":   apex_score,
            "clasificacion":clasificacion,
            "col_apex":     col_apex,
            "accion":       accion,
            "todos_pasan":  todos_pasan,
            # Filtros
            "f1_pct":f1_pct,"f1_pasa":f1_pasa,"f1_det":f1_det,
            "f2_pct":f2_pct,"f2_pasa":f2_pasa,"f2_det":f2_det,
            "f3_pct":f3_pct,"f3_pasa":f3_pasa,"f3_det":f3_det,
            # Niveles
            "entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2,"tp3":tp3,
            "trail":trail,"rr1":round(rr1,2),"rr2":round(rr2,2),
            "atr_pct":round(atr_pct,2),
            # Salidas
            "salidas":salidas,
            # Indicadores clave para display
            "rsi":round(rsi_v,1),"adx":round(adx_v,1),
            "dip":round(dip_v,1),"dim":round(dim_v,1),
            "bb_pct":round(bb_v,3),"cascada":cascada,
            "p9":round(p9,3),"p21":round(p21,3),
            "vel_kalman":round(vel_v,5),"acc_kalman":round(acc_v,5),
            "vol_ratio":round(vr_v,2),"cruce_macd":cruce_macd,
            "cruce_e21":cruce_e21,
        }
    except Exception as e:
        return None


# ── Pares APEX — los más líquidos y con mayor potencial ─────
APEX_PARES = [
    ("bitcoin",            "BTC"),
    ("ethereum",           "ETH"),
    ("solana",             "SOL"),
    ("avalanche-2",        "AVAX"),
    ("near",               "NEAR"),
    ("injective-protocol", "INJ"),
    ("aptos",              "APT"),
    ("arbitrum",           "ARB"),
    ("optimism",           "OP"),
    ("sui",                "SUI"),
    ("fetch-ai",           "FET"),
    ("chainlink",          "LINK"),
    ("uniswap",            "UNI"),
    ("aave",               "AAVE"),
    ("ripple",             "XRP"),
    ("dogecoin",           "DOGE"),
    ("cardano",            "ADA"),
    ("matic-network",      "MATIC"),
    ("near",               "NEAR"),
    ("pepe",               "PEPE"),
    ("shiba-inu",          "SHIB"),
]
# Deduplicate
seen = set()
APEX_PARES = [(cid,sym) for cid,sym in APEX_PARES
              if not (sym in seen or seen.add(sym))]

APEX_DIAS_MAP = {
    "4H · 10 días":  ("4h",  10),
    "1D · 30 días":  ("1d",  30),
    "1D · 90 días":  ("1d",  90),
}


# ══════════════════════════════════════════════════════════════
#  UI — APEX
# ══════════════════════════════════════════════════════════════
st.sidebar.divider()
st.sidebar.markdown("**🚀 AI.LINO APEX**")
with st.sidebar:
    tf_apex  = st.selectbox("⏱ TF Apex:",
                            list(APEX_DIAS_MAP.keys()),
                            index=1, key="tf_apex_sel")
    run_apex = st.button("🚀 EJECUTAR APEX",
                         use_container_width=True, key="btn_apex")

if run_apex:
    st.divider()
    # Header APEX
    st.markdown("""
    <div style="background:linear-gradient(135deg,#001a0a,#0d1428);
                border:1px solid #00ff88;border-radius:12px;
                padding:16px 20px;margin-bottom:16px;
                font-family:'Orbitron',monospace">
        <div style="color:#00ff88;font-size:1.3rem;font-weight:700;
                    letter-spacing:3px">🚀 AI.LINO APEX</div>
        <div style="color:#4488ff;font-size:0.78rem;margin-top:4px">
            SISTEMA DE ENTRADA DE ALTA PRECISIÓN · 3 FILTROS EN CASCADA
        </div>
        <div style="color:#2a4060;font-size:0.72rem;margin-top:4px">
            F1: Tendencia Base · F2: Impulso Activo · F3: Entrada Precisa
        </div>
    </div>""", unsafe_allow_html=True)

    iv_ap, d_ap = APEX_DIAS_MAP[tf_apex]

    # ── Descarga y análisis ───────────────────────────────────
    prog_ap = st.progress(0)
    resultados_ap = []
    total_ap = len(APEX_PARES)

    for i_ap,(cid,sym) in enumerate(APEX_PARES):
        prog_ap.progress((i_ap+1)/total_ap,
                        text=f"🔬 Analizando {sym} — {i_ap+1}/{total_ap}")
        try:
            df_ap = binance_descargar(sym+"USDT", iv_ap, d_ap)
            if df_ap is None or df_ap.empty:
                df_ap = coingecko_descargar(cid, d_ap)
            res_ap = apex_analizar(df_ap, sym)
            if res_ap: resultados_ap.append(res_ap)
        except:
            pass
        time.sleep(0.1)
    prog_ap.empty()

    if not resultados_ap:
        st.error("No se pudieron analizar los pares. Verifica conexión.")
    else:
        resultados_ap.sort(key=lambda x: x["apex_score"], reverse=True)

        optimas  = [r for r in resultados_ap if "ÓPTIMA" in r["clasificacion"]]
        buenas   = [r for r in resultados_ap if "BUENA"  in r["clasificacion"]]
        casi     = [r for r in resultados_ap if "CASI"   in r["clasificacion"]]
        formando = [r for r in resultados_ap if "FORMACIÓN" in r["clasificacion"]
                    or "MONITOR" in r.get("accion","")]

        st.caption(f"🔬 {len(resultados_ap)} pares analizados · "
                  f"🚀 {len(optimas)} óptimas · "
                  f"⚡ {len(buenas)} buenas · "
                  f"👀 {len(casi)} casi listas")

        # ── GRÁFICO RADAR DE SCORES ───────────────────────────
        top_vis = resultados_ap[:16]
        fig_ap  = plt.figure(figsize=(14,5), facecolor=BG)
        ax_ap   = fig_ap.add_subplot(1,1,1); estilizar_ax(ax_ap)
        x_ap    = np.arange(len(top_vis))
        w       = 0.28

        f1s = [r["f1_pct"] for r in top_vis]
        f2s = [r["f2_pct"] for r in top_vis]
        f3s = [r["f3_pct"] for r in top_vis]

        ax_ap.bar(x_ap-w,   f1s, w, color="#1D9E75", alpha=0.8, label="F1 Tendencia")
        ax_ap.bar(x_ap,     f2s, w, color="#7F77DD", alpha=0.8, label="F2 Impulso")
        ax_ap.bar(x_ap+w,   f3s, w, color="#BA7517", alpha=0.8, label="F3 Entrada")

        # Score APEX como línea
        ax_ap2 = ax_ap.twinx()
        ax_ap2.plot(x_ap, [r["apex_score"] for r in top_vis],
                   color="#00ff88", lw=2, marker="o", markersize=5,
                   label="Score APEX")
        ax_ap2.set_ylim(0,115); ax_ap2.set_ylabel("Score APEX",color="#00ff88",fontsize=8)
        ax_ap2.tick_params(colors="#00ff88", labelsize=7)
        ax_ap2.axhline(80, color="#00ff88", lw=0.8, ls="--", alpha=0.5)
        ax_ap2.axhline(65, color="#ffaa00", lw=0.8, ls=":", alpha=0.5)

        ax_ap.set_xticks(x_ap)
        ax_ap.set_xticklabels([r["sym"] for r in top_vis],
                              rotation=35, ha="right", fontsize=8, color="#4a6080")
        ax_ap.set_ylim(0,115)
        ax_ap.legend(fontsize=7, framealpha=0.3, loc="upper left")
        ax_ap.set_title("APEX — Desglose de 3 Filtros por Par",
                       color="#4488ff", fontsize=10)
        ax_ap.axhline(55, color="#ffaa00", lw=0.6, ls=":", alpha=0.4)
        plt.tight_layout(); render_fig(fig_ap)

        # ── TABS ─────────────────────────────────────────────
        tab_labels_ap = []
        grupos_ap     = []
        if optimas:  tab_labels_ap.append(f"🚀 Óptima ({len(optimas)})");  grupos_ap.append(optimas)
        if buenas:   tab_labels_ap.append(f"⚡ Buena ({len(buenas)})");    grupos_ap.append(buenas)
        if casi:     tab_labels_ap.append(f"👀 Casi ({len(casi)})");       grupos_ap.append(casi)
        if formando: tab_labels_ap.append(f"⏳ Formación ({len(formando)})");grupos_ap.append(formando)
        tab_labels_ap.append("📋 Todos")
        grupos_ap.append(resultados_ap)

        tabs_ap = st.tabs(tab_labels_ap)

        for tab_obj_ap, grupo_ap in zip(tabs_ap, grupos_ap):
            with tab_obj_ap:
                # Cards top 3
                top3_ap = grupo_ap[:3]
                if top3_ap:
                    cols_ap = st.columns(min(3, len(top3_ap)))
                    for i_c, (r, col_c) in enumerate(zip(top3_ap, cols_ap)):
                        with col_c:
                            col = r["col_apex"]
                            rr1c="#00ff88" if r["rr1"]>=2 else("#ffaa00" if r["rr1"]>=1.5 else "#ff3355")

                            # Header de la card
                            st.markdown(f"""
                            <div style="border:2px solid {col};border-radius:10px;
                                        background:#08090f;padding:12px 14px;
                                        font-family:'Share Tech Mono',monospace">
                                <div style="font-family:'Orbitron',monospace;
                                            font-size:1rem;color:{col};font-weight:700">
                                    {r['sym']}/USDT
                                </div>
                                <div style="font-size:0.68rem;color:#4488ff;margin:2px 0">
                                    {r['clasificacion']}
                                </div>
                                <div style="font-size:1.5rem;font-weight:700;color:{col}">
                                    {r['apex_score']}/100
                                </div>
                                <div style="font-size:0.85rem;color:#c0d8ff;margin:3px 0">
                                    {r['precio_fmt']}
                                </div>
                            </div>""", unsafe_allow_html=True)

                            # Barras de filtros
                            for f_lbl,f_val,f_col in [
                                ("F1 Tendencia",r["f1_pct"],"#1D9E75"),
                                ("F2 Impulso",  r["f2_pct"],"#7F77DD"),
                                ("F3 Entrada",  r["f3_pct"],"#BA7517"),
                            ]:
                                ic = "✅" if f_val>=55 else "❌"
                                st.markdown(f"""
                                <div style="margin:3px 0;font-family:'Share Tech Mono',monospace">
                                    <div style="display:flex;justify-content:space-between;
                                                font-size:0.7rem;color:#4a6060">
                                        <span>{ic} {f_lbl}</span>
                                        <span style="color:{f_col}">{f_val:.0f}/100</span>
                                    </div>
                                    <div style="background:#0d1428;border-radius:3px;height:7px;overflow:hidden">
                                        <div style="width:{f_val}%;height:100%;
                                                    background:{f_col};border-radius:3px"></div>
                                    </div>
                                </div>""", unsafe_allow_html=True)

                            # Niveles
                            st.markdown(f"""
                            <div style="margin-top:8px;font-size:0.7rem;
                                        border-top:1px solid #0d1a2e;padding-top:6px;
                                        font-family:'Share Tech Mono',monospace">
                                <div style="color:#00ff88">⚡ Entry: {r['precio_fmt']}</div>
                                <div style="color:#ff3355">🛑 SL: {fp(r['sl'])}</div>
                                <div style="color:#ffaa00">🎯 TP1: {fp(r['tp1'])} (R/R {r['rr1']}x)</div>
                                <div style="color:#ffdd44">🎯 TP2: {fp(r['tp2'])} (R/R {r['rr2']}x)</div>
                                <div style="color:#ffffff">🏆 TP3: {fp(r['tp3'])} (dejar correr)</div>
                                <div style="color:#4a6080;margin-top:4px">
                                    Trailing: {fp(r['trail'])} · ATR: {r['atr_pct']}%
                                </div>
                            </div>""", unsafe_allow_html=True)

                            # Señales de salida activas
                            if r["salidas"]:
                                for sal_tipo, sal_desc in r["salidas"]:
                                    st.error(f"{sal_tipo}: {sal_desc}")

                            # Botón agregar al Live
                            cid_btn = next((cid for cid,sym2 in APEX_PARES if sym2==r["sym"]),None)
                            if cid_btn:
                                import uuid as _uuid
                                _btn_key = f"apex_live_{r['sym']}_{str(_uuid.uuid4())[:8]}"
                                if st.button(f"🎯 SEGUIR EN VIVO — {r['sym']}", key=_btn_key,
                                            type="primary"):
                                    # Registrar posición activa con todos los datos
                                    st.session_state.posicion_activa = {
                                        "sym":      r["sym"],
                                        "cid":      cid_btn,
                                        "entry":    r["precio"],
                                        "sl":       r["sl"],
                                        "tp1":      r["tp1"],
                                        "tp2":      r["tp2"],
                                        "tp3":      r["tp3"],
                                        "trail":    r["trail"],
                                        "atr":      r["atr_pct"],
                                        "score":    r["apex_score"],
                                        "tiempo":   datetime.utcnow().isoformat(),
                                        "trail_sl": r["sl"],   # SL dinámico actual
                                        "max_precio": r["precio"],  # precio máximo visto
                                    }
                                    st.session_state.ver_seguimiento = True
                                    supa_guardar_apex_signal(
                                        r["sym"],"ENTRADA",r["precio"],
                                        r["apex_score"],r["sl"],r["tp1"],r["tp2"],
                                        str(r["f1_det"])
                                    )
                                    st.rerun()

                st.markdown("&nbsp;")

                # ── DIAGNÓSTICO DETALLADO DEL #1 ─────────────
                if grupo_ap:
                    top1_ap = grupo_ap[0]
                    with st.expander(f"🔬 Diagnóstico completo — {top1_ap['sym']}", expanded=False):
                        da1,da2,da3 = st.columns(3)

                        def render_filtro(col_d, titulo, score, det, col_f):
                            with col_d:
                                st.markdown(f"""
                                <div style="background:#08090f;border:1px solid {col_f};
                                            border-radius:8px;padding:10px 12px;
                                            font-family:'Share Tech Mono',monospace">
                                    <div style="color:{col_f};font-size:0.82rem;
                                                font-weight:700;margin-bottom:6px">
                                        {titulo} — {score:.0f}/100
                                    </div>
                                    <div style="background:#0d1428;border-radius:3px;
                                                height:8px;margin-bottom:8px;overflow:hidden">
                                        <div style="width:{score}%;height:100%;
                                                    background:{col_f};border-radius:3px"></div>
                                    </div>""", unsafe_allow_html=True)
                                for k,v in det.items():
                                    ic_col="#00ff88" if "🟢" in v else("#ffaa00" if "🟡" in v else "#ff3355")
                                    st.markdown(f"""
                                    <div style="font-size:0.72rem;margin:3px 0;
                                                display:flex;justify-content:space-between">
                                        <span style="color:#4a6080">{k}</span>
                                        <span style="color:{ic_col}">{v}</span>
                                    </div>""", unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)

                        render_filtro(da1,"F1 TENDENCIA",top1_ap["f1_pct"],
                                     top1_ap["f1_det"],"#1D9E75")
                        render_filtro(da2,"F2 IMPULSO",  top1_ap["f2_pct"],
                                     top1_ap["f2_det"],"#7F77DD")
                        render_filtro(da3,"F3 ENTRADA",  top1_ap["f3_pct"],
                                     top1_ap["f3_det"],"#BA7517")

                # ── TABLA COMPLETA ────────────────────────────
                if grupo_ap:
                    filas_ap = []
                    for r in grupo_ap:
                        filas_ap.append({
                            "Par":      r["sym"]+"/USDT",
                            "APEX":     r["apex_score"],
                            "Acción":   r["accion"],
                            "F1%":      r["f1_pct"],
                            "F2%":      r["f2_pct"],
                            "F3%":      r["f3_pct"],
                            "Entry":    r["precio_fmt"],
                            "SL":       fp(r["sl"]),
                            "TP1":      fp(r["tp1"]),
                            "TP3":      fp(r["tp3"]),
                            "R/R":      r["rr1"],
                            "ATR%":     r["atr_pct"],
                            "RSI":      r["rsi"],
                            "ADX":      r["adx"],
                            "🕯":       r["cascada"],
                            "MACD✅":  "✅" if r["cruce_macd"] else "—",
                            "E21✅":   "✅" if r["cruce_e21"]  else "—",
                            "Salidas":  len(r["salidas"]),
                        })
                    st.dataframe(pd.DataFrame(filas_ap),
                                use_container_width=True, hide_index=True,
                                column_config={
                                    "APEX": st.column_config.ProgressColumn(
                                        "APEX",min_value=0,max_value=100,format="%d"),
                                    "F1%":  st.column_config.ProgressColumn(
                                        "F1%", min_value=0,max_value=100,format="%d"),
                                    "F2%":  st.column_config.ProgressColumn(
                                        "F2%", min_value=0,max_value=100,format="%d"),
                                    "F3%":  st.column_config.ProgressColumn(
                                        "F3%", min_value=0,max_value=100,format="%d"),
                                    "R/R":  st.column_config.NumberColumn("R/R",format="%.1f"),
                                    "🕯":   st.column_config.NumberColumn("🕯",format="%d"),
                                    "Salidas":st.column_config.NumberColumn("⚠️Salidas",format="%d"),
                                })

        # ── NOTA DE RIESGO ────────────────────────────────────
        st.markdown("""
        <div style="background:#0d1428;border-left:3px solid #ffaa00;
                    border-radius:6px;padding:12px 16px;margin-top:12px;
                    font-family:'Share Tech Mono',monospace;font-size:0.77rem;color:#ffaa00">
            <b>⚠️ GESTIÓN DE RIESGO APEX:</b><br>
            · Solo entrar cuando <b>los 3 filtros pasan (F1+F2+F3 ≥ 55%)</b><br>
            · Tamaño máximo: 3-5% de capital por operación<br>
            · Respetar SL siempre — nunca moverlo en contra<br>
            · Al llegar a TP1 → mover SL a breakeven, dejar correr al TP2/TP3<br>
            · Score APEX ≥ 80 + R/R ≥ 2x = condiciones óptimas para tamaño completo<br>
            · Añadir al monitor Live para seguimiento automático
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 🎯 SEGUIMIENTO EN TIEMPO REAL — Posición Activa
#    Se activa cuando el usuario da clic en "SEGUIR EN VIVO"
#    Muestra todo lo necesario para saber CUÁNDO SALIR
# ══════════════════════════════════════════════════════════════

# Inicializar estados
if "posicion_activa" not in st.session_state:
    st.session_state.posicion_activa  = None
if "ver_seguimiento" not in st.session_state:
    st.session_state.ver_seguimiento  = False
if "historial_precios" not in st.session_state:
    st.session_state.historial_precios = []   # lista de precios capturados


@st.cache_data(ttl=30)   # refresca cada 30 segundos
def get_precio_live(coin_id):
    """Precio actual en tiempo real desde CoinGecko."""
    try:
        r = requests.get(f"{CG}/simple/price", params={
            "ids": coin_id,
            "vs_currencies": "usd",
            "include_24hr_change": "true",
            "include_1hr_change":  "true",
        }, timeout=8)
        if r.status_code == 200:
            data = r.json().get(coin_id, {})
            return {
                "precio":  data.get("usd", 0),
                "chg1h":   data.get("usd_1h_change", 0) or 0,
                "chg24h":  data.get("usd_24h_change", 0) or 0,
            }
    except:
        pass
    return None


@st.cache_data(ttl=45)
def get_ohlc_seguimiento(coin_id):
    """OHLC 2 días para calcular indicadores en tiempo real."""
    try:
        r = requests.get(f"{CG}/coins/{coin_id}/ohlc",
                        params={"vs_currency":"usd","days":2}, timeout=10)
        if r.status_code == 200 and r.json():
            df = pd.DataFrame(r.json(), columns=["ts","Open","High","Low","Close"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            df.set_index("ts", inplace=True)
            return df.astype(float)
    except:
        pass
    return None


def calcular_fuerza_actual(df_ohlc, precio_actual):
    """
    Calcula la fuerza alcista ACTUAL y los indicadores de salida.
    Retorna dict con todo lo necesario para la pantalla de seguimiento.
    """
    if df_ohlc is None or len(df_ohlc) < 8:
        return None
    try:
        c = df_ohlc["Close"].copy()
        # Reemplazar último cierre con precio actual
        c.iloc[-1] = precio_actual

        h = df_ohlc["High"]; l = df_ohlc["Low"]

        # EMAs
        e9  = c.ewm(span=9,  adjust=False).mean()
        e21 = c.ewm(span=21, adjust=False).mean()
        e50 = c.ewm(span=50, adjust=False).mean()

        # RSI
        d   = c.diff()
        g   = d.clip(lower=0).ewm(com=6, adjust=False).mean()
        ls  = (-d.clip(upper=0)).ewm(com=6, adjust=False).mean()
        rsi = float((100-(100/(1+g/(ls+1e-10)))).iloc[-1])
        if np.isnan(rsi): rsi = 50.0

        # MACD
        ef   = c.ewm(span=12, adjust=False).mean()
        es_m = c.ewm(span=26, adjust=False).mean()
        macd = ef - es_m
        msig = macd.ewm(span=9, adjust=False).mean()
        mhist= macd - msig
        mh   = float(mhist.iloc[-1]) if not np.isnan(mhist.iloc[-1]) else 0
        mh_p = float(mhist.iloc[-2]) if not np.isnan(mhist.iloc[-2]) else 0

        # Bollinger
        bb_m = c.rolling(min(10,len(c))).mean()
        bb_s = c.rolling(min(10,len(c))).std()
        bb_u = bb_m + 2*bb_s; bb_l = bb_m - 2*bb_s
        bb_p = float(((c-bb_l)/(bb_u-bb_l+1e-9)).iloc[-1])
        if np.isnan(bb_p): bb_p = 0.5

        # ADX
        tr   = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
        atr  = tr.ewm(span=7, adjust=False).mean()
        dm_p = h.diff().clip(lower=0); dm_n = (-l.diff()).clip(lower=0)
        dm_p = dm_p.where(dm_p>dm_n,0); dm_n = dm_n.where(dm_n>dm_p,0)
        a14  = tr.ewm(span=7,adjust=False).mean()
        dip  = (dm_p.ewm(span=7,adjust=False).mean()/(a14+1e-10))*100
        dim  = (dm_n.ewm(span=7,adjust=False).mean()/(a14+1e-10))*100
        dx   = ((dip-dim).abs()/(dip+dim+1e-10))*100
        adx  = float(dx.ewm(span=7,adjust=False).mean().iloc[-1])
        adx_p= float(dx.ewm(span=7,adjust=False).mean().iloc[-2]) if len(c)>2 else adx
        dip_v= float(dip.iloc[-1]); dim_v=float(dim.iloc[-1])

        # Kalman velocidad
        pk   = c.ewm(span=4, adjust=False).mean()
        vel  = float(pk.diff().iloc[-1])
        vel_p= float(pk.diff().iloc[-2]) if len(c)>2 else vel
        acel = vel > vel_p

        # Pendientes EMA
        p9  = (e9.iloc[-1]-e9.iloc[-4])/(e9.iloc[-4]+1e-10)*100  if len(c)>4 else 0
        p21 = (e21.iloc[-1]-e21.iloc[-4])/(e21.iloc[-4]+1e-10)*100 if len(c)>4 else 0

        e9v = float(e9.iloc[-1]); e21v=float(e21.iloc[-1])

        # Fuerza alcista 0-100
        fuerza = 0
        if p9  > 0: fuerza += min(25, p9*60)
        if p21 > 0: fuerza += min(20, p21*90)
        if adx > 25: fuerza += 25
        elif adx>15: fuerza += 15
        if dip_v > dim_v: fuerza += min(15,(dip_v-dim_v)*0.5)
        if vel > 0 and acel: fuerza += 10
        elif vel > 0:        fuerza += 5
        if rsi < 65:         fuerza += 5
        elif rsi > 74:       fuerza -= 15
        if bb_p > 0.85:      fuerza -= 10
        fuerza = max(0, min(100, fuerza))

        # ── SEÑALES DE SALIDA ────────────────────────────────
        señales_salida = []
        urgencia_salida = 0   # 0=nada, 1=vigilar, 2=preparar, 3=SALIR YA

        # 1. Mecha agotada — más importante
        if rsi > 74 and bb_p > 0.85:
            señales_salida.append("🚨 RSI SOBRECOMPRADO + BB TECHO")
            urgencia_salida = 3
        elif rsi > 70 and bb_p > 0.75:
            señales_salida.append("⚠️ RSI alto + BB extendido")
            urgencia_salida = max(urgencia_salida, 2)

        # 2. MACD bajista
        if mh < 0 and mh_p >= 0:
            señales_salida.append("🚨 MACD CRUCE BAJISTA")
            urgencia_salida = 3
        elif mh < mh_p and mh_p < 0:
            señales_salida.append("⚠️ MACD acelerando a la baja")
            urgencia_salida = max(urgencia_salida, 2)

        # 3. EMA cruce bajista
        if e9v < e21v:
            señales_salida.append("🚨 EMA9 CRUZÓ EMA21 ABAJO")
            urgencia_salida = 3

        # 4. ADX cayendo (tendencia debilitándose)
        if adx < adx_p and adx < 20:
            señales_salida.append("⚠️ ADX cayendo — tendencia débil")
            urgencia_salida = max(urgencia_salida, 1)

        # 5. Kalman bajista
        if vel < 0 and vel < vel_p:
            señales_salida.append("⚠️ Velocidad Kalman negativa")
            urgencia_salida = max(urgencia_salida, 1)

        # 6. Fuerza muy baja
        if fuerza < 20:
            señales_salida.append("⚠️ Fuerza alcista agotada")
            urgencia_salida = max(urgencia_salida, 2)

        return {
            "rsi": round(rsi,1), "mh": mh, "mh_p": mh_p,
            "bb_p": round(bb_p,3), "adx": round(adx,1),
            "adx_p": round(adx_p,1), "dip": round(dip_v,1),
            "dim": round(dim_v,1), "vel": vel, "acel": acel,
            "p9": round(p9,3), "p21": round(p21,3),
            "e9": e9v, "e21": e21v,
            "fuerza": round(fuerza,1),
            "señales_salida": señales_salida,
            "urgencia_salida": urgencia_salida,
            "atr_actual": float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 0,
        }
    except:
        return None


# ── PANTALLA DE SEGUIMIENTO ───────────────────────────────────
if st.session_state.get("ver_seguimiento") and st.session_state.get("posicion_activa"):
    pos = st.session_state.posicion_activa

    # ── Header de posición ──────────────────────────────────
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#001a0a,#001428);
                border:2px solid #00ff88;border-radius:12px;
                padding:14px 20px;margin-bottom:12px;
                font-family:'Orbitron',monospace">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <div>
                <span style="color:#00ff88;font-size:1.3rem;font-weight:700">
                    🎯 SEGUIMIENTO ACTIVO — {pos['sym']}/USDT
                </span>
                <div style="color:#4488ff;font-size:0.72rem;margin-top:2px">
                    Entrada: {pos['tiempo'][:16].replace('T',' ')} UTC
                </div>
            </div>
            <div style="text-align:right">
                <div style="color:#2a4060;font-size:0.68rem">Score APEX entrada</div>
                <div style="color:#00ff88;font-size:1.4rem;font-weight:700">{pos['score']}/100</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Botón cerrar posición
    col_hdr1, col_hdr2, col_hdr3 = st.columns([2,1,1])
    with col_hdr2:
        if st.button("✅ CERRÉ POSICIÓN", key="cerrar_pos",
                    help="Marca la posición como cerrada"):
            st.session_state.ver_seguimiento = False
            st.session_state.posicion_activa = None
            st.session_state.historial_precios = []
            get_precio_live.clear()
            get_ohlc_seguimiento.clear()
            st.rerun()
    with col_hdr3:
        if st.button("🔄 Actualizar", key="refresh_pos"):
            get_precio_live.clear()
            get_ohlc_seguimiento.clear()
            st.rerun()

    # ── Obtener datos actuales ──────────────────────────────
    precio_data = get_precio_live(pos["cid"])
    df_ohlc_seg = get_ohlc_seguimiento(pos["cid"])

    if precio_data is None:
        st.warning("⏳ Obteniendo datos... (CoinGecko actualiza cada 30-60 seg)")
        precio_actual = pos["entry"]
        chg1h = chg24h = 0
    else:
        precio_actual = precio_data["precio"]
        chg1h  = precio_data["chg1h"]
        chg24h = precio_data["chg24h"]

    # ── Actualizar trailing stop dinámico ───────────────────
    if precio_actual > pos.get("max_precio", pos["entry"]):
        pos["max_precio"] = precio_actual
        # Trail sube con el precio — nunca baja
        nuevo_trail_sl = precio_actual - pos["trail"]
        if nuevo_trail_sl > pos.get("trail_sl", pos["sl"]):
            pos["trail_sl"] = nuevo_trail_sl
        st.session_state.posicion_activa = pos

    # Guardar historial de precios para mini-gráfico
    hist = st.session_state.historial_precios
    if not hist or hist[-1] != precio_actual:
        hist.append(precio_actual)
        if len(hist) > 60:   # máx 60 puntos
            hist = hist[-60:]
        st.session_state.historial_precios = hist

    # ── Calcular PnL ────────────────────────────────────────
    entry      = pos["entry"]
    pnl_pct    = (precio_actual - entry) / (entry + 1e-10) * 100
    pnl_col    = "#00ff88" if pnl_pct >= 0 else "#ff3355"
    pfmt_now   = (f"${precio_actual:,.6f}" if precio_actual<1 else
                  f"${precio_actual:,.4f}" if precio_actual<10 else
                  f"${precio_actual:,.2f}")

    # Distancias a niveles
    dist_sl    = (precio_actual - pos["trail_sl"]) / precio_actual * 100
    dist_tp1   = (pos["tp1"] - precio_actual) / precio_actual * 100
    dist_tp2   = (pos["tp2"] - precio_actual) / precio_actual * 100
    dist_tp3   = (pos["tp3"] - precio_actual) / precio_actual * 100

    # TP alcanzados
    tp1_ok = precio_actual >= pos["tp1"]
    tp2_ok = precio_actual >= pos["tp2"]
    tp3_ok = precio_actual >= pos["tp3"]

    # ── INDICADORES ACTUALES ────────────────────────────────
    ind_live = calcular_fuerza_actual(df_ohlc_seg, precio_actual)
    fuerza   = ind_live["fuerza"]    if ind_live else 50
    urgencia = ind_live["urgencia_salida"] if ind_live else 0

    # ════════════════════════════════════════════════════════
    #  ALERTA DE SALIDA — si hay urgencia, mostrar arriba
    # ════════════════════════════════════════════════════════
    if urgencia == 3:
        for sig in (ind_live["señales_salida"] if ind_live else []):
            st.error(f"🚨 **SALIR AHORA** — {sig}")
        st.markdown("""
        <div style="background:#3a0000;border:2px solid #ff0000;border-radius:10px;
                    padding:14px 20px;text-align:center;
                    font-family:'Orbitron',monospace;margin-bottom:12px;
                    animation: pulse 1s infinite">
            <div style="color:#ff0000;font-size:1.2rem;font-weight:700">
                🚨 SALIR AHORA — FUERZA ALCISTA AGOTADA
            </div>
            <div style="color:#ff6666;font-size:0.8rem;margin-top:4px">
                Múltiples señales de techo detectadas. Toma tus ganancias.
            </div>
        </div>""", unsafe_allow_html=True)
    elif urgencia == 2:
        st.warning("⚠️ **PREPARARSE PARA SALIR** — señales de debilitamiento detectadas")
    elif tp1_ok and not tp2_ok:
        st.success("🎯 **TP1 ALCANZADO** — Considera mover SL a breakeven y esperar TP2")
    elif tp2_ok and not tp3_ok:
        st.success("🎯🎯 **TP2 ALCANZADO** — Excelente. Puedes cerrar la mitad y dejar correr al TP3")
    elif tp3_ok:
        st.balloons()
        st.success("🏆 **TP3 ALCANZADO — RENDIMIENTO MÁXIMO** — Considera cerrar posición completa")

    # ════════════════════════════════════════════════════════
    #  FILA 1: PRECIO + PnL + FUERZA
    # ════════════════════════════════════════════════════════
    col_p1, col_p2, col_p3 = st.columns([1,1,2])

    with col_p1:
        st.markdown(f"""
        <div style="background:#08090f;border:1px solid #1a2a4a;border-radius:10px;
                    padding:14px;font-family:'Share Tech Mono',monospace;text-align:center">
            <div style="color:#2a4060;font-size:0.68rem;text-transform:uppercase">
                Precio Actual</div>
            <div style="color:#c0d8ff;font-size:1.6rem;font-weight:700;margin:4px 0">
                {pfmt_now}</div>
            <div style="color:{'#00ff88' if chg1h>=0 else '#ff3355'};font-size:0.82rem">
                {chg1h:+.2f}% 1H</div>
            <div style="color:{'#00ff88' if chg24h>=0 else '#ff3355'};font-size:0.75rem">
                {chg24h:+.2f}% 24H</div>
        </div>""", unsafe_allow_html=True)

    with col_p2:
        st.markdown(f"""
        <div style="background:#08090f;border:1px solid {'#00ff88' if pnl_pct>=0 else '#ff3355'};
                    border-radius:10px;padding:14px;
                    font-family:'Share Tech Mono',monospace;text-align:center">
            <div style="color:#2a4060;font-size:0.68rem;text-transform:uppercase">
                PnL desde entrada</div>
            <div style="color:{pnl_col};font-size:1.8rem;font-weight:700;margin:4px 0">
                {pnl_pct:+.2f}%</div>
            <div style="color:#4a6080;font-size:0.72rem">
                Entrada: {fp(entry)}</div>
            <div style="color:#4a6080;font-size:0.72rem">
                Máx: {fp(pos.get('max_precio',entry))}</div>
        </div>""", unsafe_allow_html=True)

    with col_p3:
        # Barra de fuerza alcista — el indicador principal
        fc = "#00ff88" if fuerza>=60 else("#ffaa00" if fuerza>=35 else "#ff3355")
        estado_fuerza = ("🔥 FUERTE" if fuerza>=70 else
                        "⚡ ACTIVA" if fuerza>=50 else
                        "⚠️ DEBILITANDO" if fuerza>=30 else
                        "🔴 AGOTADA")
        st.markdown(f"""
        <div style="background:#08090f;border:1px solid {fc};border-radius:10px;
                    padding:14px;font-family:'Share Tech Mono',monospace">
            <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                <span style="color:#2a4060;font-size:0.68rem;text-transform:uppercase">
                    FUERZA ALCISTA</span>
                <span style="color:{fc};font-size:0.82rem;font-weight:700">
                    {estado_fuerza}</span>
            </div>
            <div style="background:#0d1428;border-radius:6px;height:22px;
                        overflow:hidden;position:relative;margin-bottom:6px">
                <div style="width:{fuerza}%;height:100%;background:{fc};
                            border-radius:6px;transition:width 1s"></div>
                <span style="position:absolute;right:8px;top:2px;
                             color:white;font-size:0.82rem;font-weight:700">
                    {fuerza:.0f}%</span>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;
                        gap:4px;font-size:0.7rem;color:#4a6080">
                <span>RSI: <b style="color:#ff8844">{ind_live['rsi'] if ind_live else '—'}</b></span>
                <span>ADX: <b style="color:#44aaff">{ind_live['adx'] if ind_live else '—'}</b></span>
                <span>BB: <b style="color:#ffaa00">{ind_live['bb_p'] if ind_live else '—'}</b></span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("&nbsp;")

    # ════════════════════════════════════════════════════════
    #  FILA 2: NIVELES + PENDIENTES EMA
    # ════════════════════════════════════════════════════════
    col_niv, col_ema = st.columns([1,1])

    with col_niv:
        # Niveles con semáforo visual
        def nivel_row(label, precio_niv, dist, alcanzado, col_n):
            icono = "✅" if alcanzado else "⏳"
            d_str = f"({dist:+.2f}%)" if not alcanzado else "ALCANZADO"
            return f"""<div style="display:flex;justify-content:space-between;
                        padding:5px 0;border-bottom:1px solid #0d1a2e;
                        font-size:0.78rem;font-family:'Share Tech Mono',monospace">
                <span style="color:#4a6060">{icono} {label}</span>
                <span style="color:{col_n};font-weight:700">{fp(precio_niv)}</span>
                <span style="color:{'#00ff88' if alcanzado else '#4a6060'};
                             font-size:0.7rem">{d_str}</span>
            </div>"""

        trail_sl = pos.get("trail_sl", pos["sl"])
        sl_dist  = (precio_actual - trail_sl)/precio_actual*100

        st.markdown(f"""
        <div style="background:#08090f;border:1px solid #1a2a4a;border-radius:10px;
                    padding:12px 14px;font-family:'Share Tech Mono',monospace">
            <div style="color:#4488ff;font-size:0.75rem;font-weight:700;
                        margin-bottom:8px;text-transform:uppercase">
                Niveles de Gestión</div>
            {nivel_row("SL Trailing",trail_sl,-sl_dist,False,"#ff3355")}
            {nivel_row("TP1 (50% salida)",pos["tp1"],dist_tp1,tp1_ok,"#ffaa00")}
            {nivel_row("TP2 (75% salida)",pos["tp2"],dist_tp2,tp2_ok,"#ffdd44")}
            {nivel_row("TP3 (100% salida)",pos["tp3"],dist_tp3,tp3_ok,"#ffffff")}
            <div style="margin-top:8px;font-size:0.7rem;color:#2a4060">
                🛡 Trail SL sube automáticamente con el precio<br>
                📏 ATR actual: {ind_live['atr_actual']*100/precio_actual:.2f}% del precio
            </div>
        </div>""", unsafe_allow_html=True)

    with col_ema:
        # Pendientes EMA con indicador visual
        def pend_bar(label, valor, col_bar):
            w = min(100, abs(valor)*200)
            arrow = "↑" if valor>0.02 else ("↓" if valor<-0.02 else "→")
            color = "#00ff88" if valor>0.02 else("#ff3355" if valor<-0.02 else "#ffaa00")
            return f"""<div style="margin:6px 0;font-family:'Share Tech Mono',monospace">
                <div style="display:flex;justify-content:space-between;font-size:0.72rem">
                    <span style="color:#4a6060">{label}</span>
                    <span style="color:{color};font-weight:700">{arrow} {valor:+.3f}%</span>
                </div>
                <div style="background:#0d1428;border-radius:3px;height:8px;margin:2px 0;overflow:hidden">
                    <div style="width:{w}%;height:100%;background:{color};border-radius:3px"></div>
                </div>
            </div>"""

        p9_v  = ind_live["p9"]  if ind_live else 0
        p21_v = ind_live["p21"] if ind_live else 0
        dip_v2= ind_live["dip"] if ind_live else 0
        dim_v2= ind_live["dim"] if ind_live else 0
        vel_v2= ind_live["vel"] if ind_live else 0

        # Kalman barra
        vel_pct_display = min(100, abs(vel_v2) * 50000)
        vel_col = "#00ff88" if vel_v2>0 else "#ff3355"

        st.markdown(f"""
        <div style="background:#08090f;border:1px solid #1a2a4a;border-radius:10px;
                    padding:12px 14px">
            <div style="color:#4488ff;font-size:0.75rem;font-weight:700;
                        margin-bottom:8px;text-transform:uppercase">
                Señales de Fuerza</div>
            {pend_bar("Pendiente EMA9",  p9_v,  "#00ff88")}
            {pend_bar("Pendiente EMA21", p21_v, "#ff8844")}
            <div style="margin:6px 0">
                <div style="display:flex;justify-content:space-between;font-size:0.72rem">
                    <span style="color:#4a6060">Velocidad Kalman</span>
                    <span style="color:{vel_col};font-weight:700">
                        {'↑ Acelerando' if ind_live and ind_live['acel'] else '↓ Desacelerando'}</span>
                </div>
                <div style="background:#0d1428;border-radius:3px;height:8px;margin:2px 0;overflow:hidden">
                    <div style="width:{vel_pct_display:.0f}%;height:100%;
                                background:{vel_col};border-radius:3px"></div>
                </div>
            </div>
            <div style="margin-top:8px;display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:0.72rem">
                <div style="color:#4a6060">+DI: <b style="color:#00ff88">{dip_v2:.1f}</b></div>
                <div style="color:#4a6060">-DI: <b style="color:#ff3355">{dim_v2:.1f}</b></div>
                <div style="color:#4a6060">MACD hist:
                    <b style="color:{'#00ff88' if ind_live and ind_live['mh']>0 else '#ff3355'}">
                        {'↑' if ind_live and ind_live['mh']>ind_live['mh_p'] else '↓'}
                        {ind_live['mh']:.6f if ind_live else '—'}</b></div>
                <div style="color:#4a6060">EMA9 vs 21:
                    <b style="color:{'#00ff88' if ind_live and ind_live['e9']>ind_live['e21'] else '#ff3355'}">
                        {'↑ ALCISTA' if ind_live and ind_live['e9']>ind_live['e21'] else '↓ BAJISTA'}</b></div>
            </div>
        </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    #  MINI-GRÁFICO DE PRECIO DESDE ENTRADA
    # ════════════════════════════════════════════════════════
    if len(hist) >= 2:
        st.markdown("&nbsp;")
        fig_seg = plt.figure(figsize=(14, 3), facecolor=BG)
        ax_seg  = fig_seg.add_subplot(1,1,1); estilizar_ax(ax_seg)

        x_seg = range(len(hist))
        cols_seg = ["#00ff88" if p>=entry else "#ff3355" for p in hist]

        ax_seg.fill_between(x_seg, entry, hist,
                           where=[p>=entry for p in hist],
                           alpha=0.2, color="#00ff88")
        ax_seg.fill_between(x_seg, entry, hist,
                           where=[p<entry for p in hist],
                           alpha=0.2, color="#ff3355")
        ax_seg.plot(x_seg, hist, color="#4488ff", lw=1.5, zorder=3)
        ax_seg.scatter([len(hist)-1],[hist[-1]],
                      color="#00ff88" if hist[-1]>=entry else "#ff3355",
                      s=60, zorder=5)

        # Líneas de niveles
        ax_seg.axhline(entry,              color="#ffffff", lw=1,   ls="--", alpha=0.7, label=f"Entry {fp(entry)}")
        ax_seg.axhline(pos["trail_sl"],    color="#ff3355", lw=1,   ls="--", alpha=0.8, label=f"SL {fp(pos['trail_sl'])}")
        ax_seg.axhline(pos["tp1"],         color="#ffaa00", lw=0.8, ls=":",  alpha=0.7, label=f"TP1 {fp(pos['tp1'])}")
        ax_seg.axhline(pos["tp2"],         color="#ffdd44", lw=0.8, ls=":",  alpha=0.6, label=f"TP2 {fp(pos['tp2'])}")
        ax_seg.axhline(pos["tp3"],         color="#ffffff", lw=0.7, ls=":",  alpha=0.5, label=f"TP3 {fp(pos['tp3'])}")

        ax_seg.legend(fontsize=6, framealpha=0.3, loc="upper left", ncol=5)
        ax_seg.set_title(f"Precio {pos['sym']} desde entrada — {len(hist)} puntos capturados",
                        color="#4488ff", fontsize=9)
        ax_seg.set_xlim(0, max(1,len(hist)-1))
        ax_seg.set_xticks([])
        plt.tight_layout(); render_fig(fig_seg)

    # ════════════════════════════════════════════════════════
    #  SEÑALES DE SALIDA ACTIVAS
    # ════════════════════════════════════════════════════════
    if ind_live and ind_live["señales_salida"]:
        st.markdown("### ⚠️ Señales detectadas")
        for sig in ind_live["señales_salida"]:
            if "🚨" in sig:
                st.error(sig)
            else:
                st.warning(sig)
    elif ind_live:
        st.success("✅ Sin señales de salida — la fuerza alcista se mantiene")

    # ── Auto-refresh ──────────────────────────────────────
    st.markdown("---")
    ref1, ref2, ref3 = st.columns([2,1,1])
    ref1.caption(f"⏱ Actualización automática cada 60 seg · "
                f"Puntos capturados: {len(hist)} · "
                f"Máximo precio visto: {fp(pos.get('max_precio',entry))}")
    if ref2.button("🔄 Forzar update", key="force_refresh_seg"):
        get_precio_live.clear()
        get_ohlc_seguimiento.clear()
        st.rerun()
    if ref3.button("❌ Cancelar seguimiento", key="cancel_seg"):
        st.session_state.ver_seguimiento  = False
        st.session_state.posicion_activa  = None
        st.session_state.historial_precios= []
        st.rerun()

    # Auto-refresh cada 60 segundos
    time.sleep(60)
    get_precio_live.clear()
    st.rerun()
