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
# 🔥 SCANNER BITSO — Detecta subidas ACTIVAS ahora mismo
#    Filosofía: rápido, visual, práctico.
#    No teoría — lo que VES en la pantalla de Bitso.
# ══════════════════════════════════════════════════════════════

# Pares disponibles en Bitso (o equivalentes en Binance para datos)
PARES_BITSO = [
    "BTCUSDT","ETHUSDT","XRPUSDT","LTCUSDT","BCHUSDT",
    "LINKUSDT","BATUSDT","MANAУСDT","MATICUSDT","SOLUSDT",
    "DOGEUSDT","USDTUSDT","ADAUSDT","BNBUSDT","AVAXUSDT",
    "DOTUSDT","UNIUSDT","AAVEUSDT","SHIBUSDT","NEARUSDT",
    "ATOMUSDT","FTMUSDT","SANDUSDT","AXSUSDT","CRVUSDT",
    "MKRUSDT","COMPUSDT","SNXUSDT","INJUSDT","APTUSDT",
    "ARBUSDT","OPUSDT","SUIUSDT","FETUSDT","WLDUSDT",
]
# Limpiar typos
PARES_BITSO = [p for p in PARES_BITSO if p.isascii() and p.endswith("USDT")]


def analizar_par_bitso(df, sym):
    """
    Análisis práctico enfocado en:
    1. ¿Está subiendo AHORA? (cascada de velas alcistas)
    2. ¿Qué tan estable es la subida? (ATR%, volatilidad relativa)
    3. ¿Le queda mecha? (RSI, posición Bollinger, volumen)
    4. ¿Hay fuerza real? (momentum, EMA alineación)
    """
    if df is None or df.empty or len(df) < 20:
        return None
    try:
        c = df["Close"]; o = df["Open"]
        h = df["High"];  l = df["Low"]
        v = df["Volume"]
        n = len(c)

        # ── EMAs ───────────────────────────────────────────
        e9  = c.ewm(span=9,  adjust=False).mean()
        e21 = c.ewm(span=21, adjust=False).mean()
        e50 = c.ewm(span=50, adjust=False).mean()

        # ── RSI ────────────────────────────────────────────
        d   = c.diff()
        g   = d.clip(lower=0).ewm(com=13, adjust=False).mean()
        ls  = (-d.clip(upper=0)).ewm(com=13, adjust=False).mean()
        rsi = (100-(100/(1+g/(ls+1e-10)))).iloc[-1]

        # ── ATR% — estabilidad ────────────────────────────
        tr  = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
        atr = tr.ewm(span=14, adjust=False).mean()
        atr_pct = (atr / c * 100).iloc[-1]   # % del precio → más bajo = más estable

        # ── Bollinger ─────────────────────────────────────
        bb_m = c.rolling(20).mean()
        bb_s = c.rolling(20).std()
        bb_u = bb_m + 2*bb_s
        bb_l = bb_m - 2*bb_s
        bb_pct = ((c - bb_l)/(bb_u - bb_l + 1e-9)).iloc[-1]

        # ── Volumen ───────────────────────────────────────
        vol_sma   = v.rolling(20).mean()
        vol_ratio = (v / (vol_sma + 1e-10)).iloc[-1]

        # ══════════════════════════════════════════════════
        #  CASCADA ALCISTA — cuántas velas consecutivas
        #  están subiendo (cierre > apertura)
        # ══════════════════════════════════════════════════
        cascada = 0
        for i in range(1, min(12, n)):
            if c.iloc[-i] > o.iloc[-i]:   # vela verde
                cascada += 1
            else:
                break

        # Retornos recientes
        ret1  = (c.iloc[-1] - c.iloc[-2])  / (c.iloc[-2]+1e-10)*100
        ret3  = (c.iloc[-1] - c.iloc[-4])  / (c.iloc[-4]+1e-10)*100 if n>4 else 0
        ret6  = (c.iloc[-1] - c.iloc[-7])  / (c.iloc[-7]+1e-10)*100 if n>7 else 0
        ret12 = (c.iloc[-1] - c.iloc[-13]) / (c.iloc[-13]+1e-10)*100 if n>13 else 0

        precio_actual = c.iloc[-1]

        # ══════════════════════════════════════════════════
        #  MECHA RESTANTE — cuánto le queda antes de agotarse
        # ══════════════════════════════════════════════════
        # Combina RSI + BB posición + pendiente EMA
        # 100 = mucha mecha (inicio), 0 = sin mecha (fin)
        mecha_rsi  = max(0, min(100, (75 - rsi) / 45 * 100)) if not pd.isna(rsi) else 50
        mecha_bb   = max(0, min(100, (1 - bb_pct) * 100))    if not pd.isna(bb_pct) else 50
        # Pendiente EMA9 — si acelera tiene más mecha
        e9_arr     = e9.values
        pendiente  = (e9_arr[-1]-e9_arr[-4])/(e9_arr[-4]+1e-10)*100 if n>4 else 0
        mecha_pend = min(100, max(0, 50 + pendiente*5))

        mecha_total = (mecha_rsi*0.4 + mecha_bb*0.35 + mecha_pend*0.25)

        # ══════════════════════════════════════════════════
        #  ESTABILIDAD DE LA SUBIDA
        #  ATR% bajo + retornos consistentes = subida estable
        # ══════════════════════════════════════════════════
        # Retornos de las últimas 6 velas
        rets_6 = [
            (c.iloc[-i] - c.iloc[-i-1])/(c.iloc[-i-1]+1e-10)*100
            for i in range(1, min(7, n))
        ]
        if rets_6:
            consistencia = sum(1 for r in rets_6 if r > 0) / len(rets_6) * 100
            volatilidad_rets = np.std(rets_6) if len(rets_6)>1 else 0
        else:
            consistencia = 50; volatilidad_rets = 0

        # Estabilidad: alta si ATR% bajo Y retornos consistentes
        estabilidad = max(0, min(100,
            (1 - min(atr_pct, 10)/10) * 50 +          # 50% peso ATR%
            consistencia * 0.3 +                         # 30% consistencia
            max(0, 20 - volatilidad_rets*2) * 1.0       # 20% baja volatilidad
        ))

        # ══════════════════════════════════════════════════
        #  SCORE GLOBAL DE OPORTUNIDAD
        # ══════════════════════════════════════════════════
        score = 0

        # ── Cascada (0-30) ─────────────────────────────
        if cascada >= 5:   score += 30
        elif cascada >= 3: score += 22
        elif cascada >= 2: score += 14
        elif cascada >= 1: score += 7
        # Sin cascada = no está subiendo ahora

        # ── Mecha restante (0-25) ──────────────────────
        score += int(mecha_total * 0.25)

        # ── Estabilidad (0-20) ────────────────────────
        score += int(estabilidad * 0.20)

        # ── Momentum reciente (0-15) ──────────────────
        if ret6 > 5:   score += 15
        elif ret6 > 2: score += 10
        elif ret6 > 0: score += 5

        # ── Volumen confirma (0-10) ───────────────────
        if vol_ratio > 2:   score += 10
        elif vol_ratio > 1.3: score += 6
        elif vol_ratio > 0.8: score += 3

        # Penalizaciones
        if rsi > 78:           score -= 15   # sobrecomprado
        if bb_pct > 0.92:      score -= 10   # sobre banda superior
        if cascada == 0:       score -= 20   # no está subiendo
        if ret3 < 0:           score -= 10   # bajó en últimas 3 velas

        score = max(0, min(100, score))

        # ── Clasificación ─────────────────────────────
        if score >= 72 and cascada >= 3 and mecha_total > 40:
            estado = "🔥 SUBIENDO FUERTE"
            estado_col = "#00ff88"
        elif score >= 55 and cascada >= 2:
            estado = "⚡ SUBIDA ACTIVA"
            estado_col = "#44ffaa"
        elif score >= 40 and cascada >= 1:
            estado = "📈 SUBIENDO"
            estado_col = "#ffaa00"
        elif cascada == 0 and ret3 < 0:
            estado = "📉 BAJANDO"
            estado_col = "#ff3355"
        else:
            estado = "↔️ LATERAL"
            estado_col = "#4488ff"

        # ── Nivel de mecha visual ─────────────────────
        if mecha_total >= 70:   mecha_estado = "🟢 MUCHA"
        elif mecha_total >= 45: mecha_estado = "🟡 MEDIA"
        elif mecha_total >= 25: mecha_estado = "🟠 POCA"
        else:                   mecha_estado = "🔴 AGOTADA"

        # ── ATR estabilidad texto ─────────────────────
        if atr_pct < 1.5:    atr_est = "🟢 MUY ESTABLE"
        elif atr_pct < 3:    atr_est = "🟡 ESTABLE"
        elif atr_pct < 6:    atr_est = "🟠 VOLATIL"
        else:                atr_est = "🔴 MUY VOLATIL"

        return {
            "symbol":       sym,
            "score":        score,
            "estado":       estado,
            "estado_col":   estado_col,
            "precio":       precio_actual,
            "cascada":      cascada,
            "ret1":         round(ret1,  2),
            "ret3":         round(ret3,  2),
            "ret6":         round(ret6,  2),
            "ret12":        round(ret12, 2),
            "rsi":          round(rsi,   1) if not pd.isna(rsi) else 50,
            "atr_pct":      round(atr_pct,2) if not pd.isna(atr_pct) else 0,
            "atr_est":      atr_est,
            "bb_pct":       round(bb_pct,3) if not pd.isna(bb_pct) else 0.5,
            "vol_ratio":    round(vol_ratio,2) if not pd.isna(vol_ratio) else 1,
            "mecha_total":  round(mecha_total, 1),
            "mecha_estado": mecha_estado,
            "estabilidad":  round(estabilidad, 1),
            "consistencia": round(consistencia, 1),
        }
    except:
        return None


# ── UI SCANNER BITSO ───────────────────────────────────────
st.sidebar.divider()
st.sidebar.markdown("**🔥 SCANNER BITSO**")

with st.sidebar:
    tf_bitso = st.selectbox("⏱ Timeframe:",
        ["1H · 3 días","4H · 10 días","1D · 30 días"],
        key="tf_bitso", index=0)
    run_bitso = st.button("🔥 ESCANEAR AHORA", use_container_width=True, key="btn_bitso")

if run_bitso:
    st.divider()
    st.markdown("## 🔥 SCANNER BITSO — Subidas Activas")

    tf_b_map = {
        "1H · 3 días":   ("1h", 3),
        "4H · 10 días":  ("4h", 10),
        "1D · 30 días":  ("1d", 30),
    }
    iv_b, d_b = tf_b_map[tf_bitso]

    prog_b = st.progress(0)
    resultados_b = []
    total_b = len(PARES_BITSO)

    for i_b, sym_b in enumerate(PARES_BITSO):
        prog_b.progress((i_b+1)/total_b, text=f"📡 {sym_b.replace('USDT','')} ({i_b+1}/{total_b})")
        try:
            df_b = binance_descargar(sym_b, iv_b, d_b)
            res_b = analizar_par_bitso(df_b, sym_b)
            if res_b: resultados_b.append(res_b)
        except:
            pass
        time.sleep(0.05)
    prog_b.empty()

    if not resultados_b:
        st.error("No se pudieron obtener datos. Verifica conexión.")
    else:
        # Ordenar por score
        resultados_b.sort(key=lambda x: x["score"], reverse=True)

        subiendo = [r for r in resultados_b if "SUBIENDO" in r["estado"] or "FUERTE" in r["estado"]]
        lateral  = [r for r in resultados_b if "LATERAL" in r["estado"]]
        bajando  = [r for r in resultados_b if "BAJANDO" in r["estado"]]

        st.caption(f"📊 {len(resultados_b)} pares · 🔥 {len(subiendo)} subiendo · ↔️ {len(lateral)} lateral · 📉 {len(bajando)} bajando")

        # ══════════════════════════════════════════════════
        #  MAPA DE CALOR VISUAL — un vistazo a todo el mercado
        # ══════════════════════════════════════════════════
        fig_mapa = plt.figure(figsize=(14, 6), facecolor=BG)
        ax_m = fig_mapa.add_subplot(1,1,1); estilizar_ax(ax_m)

        nombres = [r["symbol"].replace("USDT","") for r in resultados_b]
        scores  = [r["score"] for r in resultados_b]
        mechas  = [r["mecha_total"] for r in resultados_b]
        cascadas= [r["cascada"] for r in resultados_b]

        x = np.arange(len(nombres))
        # Barras principales — score
        colores_m = []
        for r in resultados_b:
            if r["score"] >= 72:   colores_m.append("#00ff88")
            elif r["score"] >= 55: colores_m.append("#44ffaa")
            elif r["score"] >= 40: colores_m.append("#ffaa00")
            elif "BAJANDO" in r["estado"]: colores_m.append("#ff3355")
            else: colores_m.append("#334466")

        bars_m = ax_m.bar(x, scores, color=colores_m, alpha=0.85, width=0.7, zorder=3)

        # Marcador de cascada (número de velas verdes encima)
        for xi, r in enumerate(resultados_b):
            if r["cascada"] > 0:
                ax_m.text(xi, r["score"]+1.5, f"🕯{r['cascada']}",
                         ha="center", va="bottom", fontsize=7, color="white")
            if r["score"] >= 55:
                ax_m.text(xi, r["score"]/2, f"{r['score']}",
                         ha="center", va="center", fontsize=7,
                         color="black", fontweight="bold")

        ax_m.axhline(72, color="#00ff88", lw=1, ls="--", alpha=0.6)
        ax_m.axhline(55, color="#ffaa00", lw=0.8, ls=":", alpha=0.5)
        ax_m.set_xticks(x)
        ax_m.set_xticklabels(nombres, rotation=40, ha="right", fontsize=7, color="#4a6080")
        ax_m.set_ylim(0, 115)
        ax_m.set_ylabel("Score", color="#2a4060", fontsize=8)
        ax_m.set_title(f"Mapa del Mercado — {tf_bitso} · 🕯N = velas verdes consecutivas",
                      color="#4488ff", fontsize=9)
        ax_m.grid(axis="y", color="#0d1a2e", lw=0.5, alpha=0.4)
        plt.tight_layout(); render_fig(fig_mapa)

        # ══════════════════════════════════════════════════
        #  TABS PRINCIPALES
        # ══════════════════════════════════════════════════
        tab_sub, tab_atr, tab_mecha, tab_todos = st.tabs([
            f"🔥 Subiendo ({len(subiendo)})",
            "📏 Más Estables (ATR)",
            "🕯 Mecha Restante",
            "📋 Todos"
        ])

        # ── TAB 1: SUBIENDO AHORA ─────────────────────────
        with tab_sub:
            if not subiendo:
                st.info("No hay pares claramente alcistas ahora en este timeframe. Prueba 1H.")
            else:
                st.markdown("**Pares con subida activa — ordenados por fuerza:**")
                # Cards en grid 3 columnas
                for i_row in range(0, min(9, len(subiendo)), 3):
                    fila = subiendo[i_row:i_row+3]
                    cols_f = st.columns(len(fila))
                    for col_f, r in zip(cols_f, fila):
                        pf   = r["precio"]
                        pfmt = f"${pf:,.6f}" if pf<1 else f"${pf:,.4f}" if pf<10 else f"${pf:,.2f}"
                        # Barra de cascada visual
                        velas_html = "🟩" * r["cascada"] + "⬜" * max(0, 6-r["cascada"])
                        col_f.markdown(f"""
                        <div style="background:#08090f;border:1.5px solid {r['estado_col']};
                                    border-radius:10px;padding:12px 14px;
                                    font-family:'Share Tech Mono',monospace;margin-bottom:8px">
                            <div style="font-family:'Orbitron',monospace;font-size:0.95rem;
                                        color:{r['estado_col']};font-weight:700">
                                {r['symbol'].replace('USDT','/USDT')}
                            </div>
                            <div style="font-size:0.68rem;color:#4488ff">{r['estado']}</div>
                            <div style="font-size:1.4rem;font-weight:700;color:{r['estado_col']};margin:3px 0">
                                {pfmt}
                                <span style="font-size:0.8rem;color:{'#00ff88' if r['ret1']>0 else '#ff3355'}">
                                    {r['ret1']:+.2f}%
                                </span>
                            </div>
                            <div style="font-size:1rem;margin:4px 0" title="Velas verdes consecutivas">
                                {velas_html}
                                <span style="font-size:0.72rem;color:#4a6080"> {r['cascada']} velas</span>
                            </div>
                            <div style="display:grid;grid-template-columns:1fr 1fr;gap:3px;margin-top:6px;font-size:0.7rem">
                                <div><span style="color:#2a4060">Mecha:</span>
                                     <span style="color:{r['estado_col']}">{r['mecha_estado']}</span></div>
                                <div><span style="color:#2a4060">ATR:</span>
                                     <span style="color:#ffaa00">{r['atr_pct']}%</span></div>
                                <div><span style="color:#2a4060">RSI:</span>
                                     <span style="color:#ff8844">{r['rsi']}</span></div>
                                <div><span style="color:#2a4060">Vol:</span>
                                     <span style="color:#44aaff">{r['vol_ratio']}x</span></div>
                                <div><span style="color:#2a4060">6v:</span>
                                     <span style="color:{'#00ff88' if r['ret6']>0 else '#ff3355'}">{r['ret6']:+.1f}%</span></div>
                                <div><span style="color:#2a4060">Score:</span>
                                     <span style="color:{r['estado_col']};font-weight:700">{r['score']}/100</span></div>
                            </div>
                        </div>""", unsafe_allow_html=True)

        # ── TAB 2: MÁS ESTABLES (menor ATR%) ─────────────
        with tab_atr:
            st.markdown("**Criptos con subida más estable (menor volatilidad — ATR% más bajo):**")
            st.caption("Un ATR% bajo significa que la subida es ordenada y controlada, no un pump salvaje.")

            # Solo los que están subiendo, ordenados por ATR% (menor = mejor)
            subiendo_estables = sorted(
                [r for r in resultados_b if r["score"] >= 40 and r["cascada"] >= 1],
                key=lambda x: x["atr_pct"]
            )

            if not subiendo_estables:
                st.info("No hay pares en subida estable ahora.")
            else:
                # Gráfico ATR% vs Score — scatter plot
                fig_atr = plt.figure(figsize=(14, 5), facecolor=BG)
                ax_atr = fig_atr.add_subplot(1,1,1); estilizar_ax(ax_atr)

                for r in subiendo_estables:
                    col_atr = r["estado_col"]
                    size_atr = r["score"] * 3
                    ax_atr.scatter(r["atr_pct"], r["score"],
                                  s=size_atr, color=col_atr, alpha=0.8, zorder=3)
                    ax_atr.annotate(r["symbol"].replace("USDT",""),
                                   (r["atr_pct"], r["score"]),
                                   textcoords="offset points", xytext=(4,4),
                                   fontsize=7, color="#c0d8ff")

                ax_atr.axvline(3, color="#ffaa00", lw=1, ls="--", alpha=0.5, label="ATR 3% (límite estable)")
                ax_atr.axhline(60, color="#00ff88", lw=0.8, ls=":", alpha=0.5, label="Score 60")
                ax_atr.set_xlabel("ATR% (← más estable | más volátil →)", color="#2a4060", fontsize=8)
                ax_atr.set_ylabel("Score", color="#2a4060", fontsize=8)
                ax_atr.set_title("Estabilidad vs Fuerza — Ideal: arriba izquierda (alto score, bajo ATR)",
                                color="#4488ff", fontsize=9)
                ax_atr.legend(fontsize=7, framealpha=0.3)
                plt.tight_layout(); render_fig(fig_atr)

                # Tabla ordenada por ATR%
                filas_atr = []
                for r in subiendo_estables[:15]:
                    pf = r["precio"]
                    pfmt = f"${pf:,.6f}" if pf<1 else f"${pf:,.4f}" if pf<10 else f"${pf:,.2f}"
                    filas_atr.append({
                        "Par":          r["symbol"].replace("USDT","/USDT"),
                        "ATR%":         r["atr_pct"],
                        "Estabilidad":  r["atr_est"],
                        "Score":        r["score"],
                        "Cascada 🕯":   r["cascada"],
                        "Mecha":        r["mecha_estado"],
                        "Precio":       pfmt,
                        "Ret 6v":      f"{r['ret6']:+.1f}%",
                        "RSI":          r["rsi"],
                        "Vol Ratio":   f"{r['vol_ratio']}x",
                    })
                if filas_atr:
                    st.dataframe(pd.DataFrame(filas_atr), use_container_width=True, hide_index=True,
                                column_config={
                                    "ATR%":      st.column_config.NumberColumn("ATR%", format="%.2f%%"),
                                    "Score":     st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
                                    "Cascada 🕯":st.column_config.NumberColumn("🕯 Velas", format="%d"),
                                })

        # ── TAB 3: MECHA RESTANTE ─────────────────────────
        with tab_mecha:
            st.markdown("**¿Cuánto le queda a cada subida? (RSI + BB posición + momentum)**")
            st.caption("🟢 Mucha mecha = puede seguir subiendo. 🔴 Agotada = probable reversión próxima.")

            # Todos ordenados por mecha
            con_mecha = sorted(
                [r for r in resultados_b if r["cascada"] >= 1],
                key=lambda x: x["mecha_total"], reverse=True
            )

            if not con_mecha:
                st.info("No hay pares en movimiento para evaluar mecha.")
            else:
                # Gráfico de mecha horizontal
                fig_mec = plt.figure(figsize=(14, max(5, len(con_mecha)*0.35)), facecolor=BG)
                ax_mec  = fig_mec.add_subplot(1,1,1); estilizar_ax(ax_mec)

                syms_mec   = [r["symbol"].replace("USDT","") for r in con_mecha]
                mechas_mec = [r["mecha_total"] for r in con_mecha]
                cols_mec   = []
                for m in mechas_mec:
                    if m >= 70:   cols_mec.append("#00ff88")
                    elif m >= 45: cols_mec.append("#ffaa00")
                    elif m >= 25: cols_mec.append("#ff8844")
                    else:         cols_mec.append("#ff3355")

                y_pos = np.arange(len(syms_mec))
                bars_mec = ax_mec.barh(y_pos, mechas_mec, color=cols_mec, alpha=0.85, height=0.65)
                ax_mec.set_yticks(y_pos); ax_mec.set_yticklabels(syms_mec, fontsize=7, color="#4a6080")

                for bar, r in zip(bars_mec, con_mecha):
                    ax_mec.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                               f"{r['mecha_total']:.0f}% · RSI:{r['rsi']:.0f}",
                               va="center", fontsize=6.5, color="white")

                ax_mec.axvline(70, color="#00ff88", lw=1, ls="--", alpha=0.6, label="Mucha mecha")
                ax_mec.axvline(25, color="#ff3355", lw=1, ls="--", alpha=0.6, label="Poca mecha")
                ax_mec.set_xlim(0, 115)
                ax_mec.set_title("Mecha Restante por Par (→ mucha mecha = puede seguir subiendo)",
                                color="#4488ff", fontsize=9)
                ax_mec.legend(fontsize=7, framealpha=0.3)
                plt.tight_layout(); render_fig(fig_mec)

                # Tabla
                filas_mec = []
                for r in con_mecha:
                    pf   = r["precio"]
                    pfmt = f"${pf:,.6f}" if pf<1 else f"${pf:,.4f}" if pf<10 else f"${pf:,.2f}"
                    filas_mec.append({
                        "Par":        r["symbol"].replace("USDT","/USDT"),
                        "Mecha %":    r["mecha_total"],
                        "Estado":     r["mecha_estado"],
                        "RSI":        r["rsi"],
                        "BB %B":      r["bb_pct"],
                        "Cascada🕯":  r["cascada"],
                        "Score":      r["score"],
                        "Precio":     pfmt,
                        "Ret 1v":    f"{r['ret1']:+.2f}%",
                        "Ret 6v":    f"{r['ret6']:+.1f}%",
                    })
                st.dataframe(pd.DataFrame(filas_mec), use_container_width=True, hide_index=True,
                            column_config={
                                "Mecha %":  st.column_config.ProgressColumn("Mecha %", min_value=0, max_value=100, format="%.0f"),
                                "Score":    st.column_config.ProgressColumn("Score",   min_value=0, max_value=100, format="%d"),
                                "Cascada🕯":st.column_config.NumberColumn("🕯 Velas",  format="%d"),
                            })

        # ── TAB 4: TODOS ──────────────────────────────────
        with tab_todos:
            filas_t = []
            for r in resultados_b:
                pf   = r["precio"]
                pfmt = f"${pf:,.6f}" if pf<1 else f"${pf:,.4f}" if pf<10 else f"${pf:,.2f}"
                filas_t.append({
                    "Par":       r["symbol"].replace("USDT","/USDT"),
                    "Estado":    r["estado"],
                    "Score":     r["score"],
                    "🕯 Velas":  r["cascada"],
                    "Mecha":     r["mecha_total"],
                    "ATR%":      r["atr_pct"],
                    "RSI":       r["rsi"],
                    "Ret 1v":   f"{r['ret1']:+.2f}%",
                    "Ret 3v":   f"{r['ret3']:+.2f}%",
                    "Ret 6v":   f"{r['ret6']:+.1f}%",
                    "Vol":      f"{r['vol_ratio']}x",
                    "Precio":    pfmt,
                })
            st.dataframe(pd.DataFrame(filas_t), use_container_width=True, hide_index=True,
                        column_config={
                            "Score":    st.column_config.ProgressColumn("Score",  min_value=0, max_value=100, format="%d"),
                            "Mecha":    st.column_config.ProgressColumn("Mecha%", min_value=0, max_value=100, format="%.0f"),
                            "🕯 Velas": st.column_config.NumberColumn("🕯 Velas", format="%d"),
                            "ATR%":     st.column_config.NumberColumn("ATR%", format="%.2f"),
                        })

        # ── SELECCIÓN MANUAL → ANALIZAR ───────────────────
        st.divider()
        st.markdown("### 🎯 Analizar par específico del scanner")
        st.caption("Selecciona cualquier par del resultado y ábrelo en el motor principal.")
        opciones_sel = [r["symbol"].replace("USDT","/USDT") + f" — Score {r['score']} · {r['estado']}"
                       for r in resultados_b]
        sel_par = st.selectbox("Elige un par para análisis completo:", opciones_sel, key="sel_par_scanner")
        if sel_par:
            sym_sel = sel_par.split(" —")[0].replace("/USDT","USDT").replace("/","")
            st.info(f"✅ Seleccionado: **{sym_sel}** — Ve al sidebar, busca este ticker y presiona **⚛️ ANALIZAR v4** para el análisis completo con HMM, Kalman y módulos cuánticos.")
