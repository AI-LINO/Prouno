# ╔══════════════════════════════════════════════════════════════╗
# ║          AI.LINO QUANTUM ENGINE v4.0                        ║
# ║  Filtro Tendencia · Entradas Precisas · Trailing · MTF     ║
# ║  Backtesting · Confluence Score · HMM 5D                   ║
# ╚══════════════════════════════════════════════════════════════╝
1import streamlit as st
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
# 🏆 SCANNER DE SUBIDAS SOSTENIDAS (Wyckoff + Smart Money)
#
# Filosofía: No buscamos el primer tick de subida.
# Buscamos criptos con ESTRUCTURA SÓLIDA que garantizan
# subidas de días/semanas, no de horas.
#
# MODELO DE 5 CAPAS:
# L1 - Estructura Wyckoff    → ¿Hubo acumulación previa sólida?
# L2 - Smart Money / Volumen → ¿Están entrando capitales reales?
# L3 - Momentum Técnico      → ¿RSI/MACD/EMAs confirman?
# L4 - Fuerza Relativa       → ¿Esta cripto es más fuerte que BTC?
# L5 - Kalman Trend          → ¿La tendencia filtrada es alcista?
# ══════════════════════════════════════════════════════════════

# Pares a escanear — priorizamos liquidez media-alta
SCAN_PARES_SOSTENIDO = [
    # Layer 1 / Infra
    "SOLUSDT","AVAXUSDT","NEARUSDT","APTUSDT","SUIUSDT","INJUSDT",
    "SEIUSDT","TIAUSDT","FETUSDT","RENDERUSDT","WLDUSDT",
    # DeFi
    "UNIUSDT","AAVEUSDT","MKRUSDT","GMXUSDT","JUPUSDT","RUNEUSDT",
    # Layer 2
    "ARBUSDT","OPUSDT","IMXUSDT","STRKUSDT","METISUSDT",
    # Ecosistemas fuertes
    "LINKUSDT","DOTUSDT","ATOMUSDT","ADAUSDT","MATICUSDT",
    # Narrativas AI/DePIN
    "AGIXUSDT","RNDRУСDT","ARKMUSDT","TAOУСDT","AIUSDT",
    # BTC ecosystem
    "BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","LTCUSDT",
    # Mid caps momentum
    "DYMUSDT","ALTUSDT","MANTAUSDT","EIGENUSDT","STXUSDT",
    "OMUSDT","ENAUSDT","PENDLEUSDT","PYTHUSDT","WUSDT",
]

# Pares con errores frecuentes en Binance (excluir)
EXCLUIR = {"RNDRУСDT","ТAOУСDT"}  # typos con caracteres cirílicos
SCAN_PARES_SOSTENIDO = [p for p in SCAN_PARES_SOSTENIDO if p not in EXCLUIR and p.isascii()]


def detectar_wyckoff_acumulacion(df):
    """
    L1 — Detecta si hubo acumulación Wyckoff antes del movimiento actual.
    Señales de acumulación:
    - Precio lateral 10-40 velas con contracción de rango
    - Volumen decreciente durante la lateral (spring)
    - Mínimos que no bajan (soporte)
    - Luego ruptura con volumen (Sign of Strength)
    Retorna score 0-30 + descripción
    """
    if len(df) < 40:
        return 0, "N/D"
    c   = df["Close"].values
    v   = df["Volume"].values
    h   = df["High"].values
    l   = df["Low"].values
    n   = len(c)

    score_w = 0
    desc_w  = []

    # Ventana de análisis: últimas 35 velas ANTES de las 5 más recientes
    # (queremos ver si ANTES de la subida hubo base)
    base_start = max(0, n - 40)
    base_end   = max(5, n - 5)
    base_c     = c[base_start:base_end]
    base_v     = v[base_start:base_end]
    base_h     = h[base_start:base_end]
    base_l     = l[base_start:base_end]

    if len(base_c) < 10:
        return 0, "Pocos datos"

    # ── Criterio 1: Rango de precios comprimido en la base ──
    rango_base   = (np.max(base_h) - np.min(base_l)) / (np.mean(base_c) + 1e-10) * 100
    rango_actual = (np.max(h[-10:]) - np.min(l[-10:])) / (np.mean(c[-10:]) + 1e-10) * 100

    if rango_base < 15:
        score_w += 10; desc_w.append("Consolidación estrecha")
    elif rango_base < 25:
        score_w += 6;  desc_w.append("Consolidación moderada")
    else:
        desc_w.append("Sin consolidación previa")

    # ── Criterio 2: Volumen decreciente en la base (acumulación silenciosa) ──
    if len(base_v) > 6:
        vol_primera_mitad = np.mean(base_v[:len(base_v)//2])
        vol_segunda_mitad = np.mean(base_v[len(base_v)//2:])
        if vol_segunda_mitad < vol_primera_mitad * 0.85:
            score_w += 8; desc_w.append("Vol. decreciente en base")
        elif vol_segunda_mitad < vol_primera_mitad:
            score_w += 4; desc_w.append("Vol. ligeramente bajo")

    # ── Criterio 3: Mínimos que se sostienen (soporte firme) ──
    lows_base = base_l[-10:]
    if len(lows_base) > 3:
        # Los últimos mínimos no deben caer más del 3% vs el primero
        lows_drift = (np.min(lows_base[-5:]) - np.min(lows_base[:5])) / (np.min(lows_base[:5]) + 1e-10) * 100
        if lows_drift > -3:
            score_w += 7; desc_w.append("Soporte firme")
        elif lows_drift > -8:
            score_w += 3; desc_w.append("Soporte débil")

    # ── Criterio 4: Sign of Strength — vela grande con volumen en ruptura ──
    # La última vela grande alcista con volumen alto
    recent_retorno = (c[-1] - c[-6]) / (c[-6] + 1e-10) * 100
    vol_reciente   = np.mean(v[-3:]) / (np.mean(v[:-3]) + 1e-10)

    if recent_retorno > 3 and vol_reciente > 1.5:
        score_w += 5; desc_w.append("Sign of Strength ✅")
    elif recent_retorno > 1.5 and vol_reciente > 1.2:
        score_w += 2; desc_w.append("Ruptura moderada")

    desc_final = " · ".join(desc_w) if desc_w else "Sin patrón claro"
    return min(score_w, 30), desc_final


def detectar_smart_money_volumen(df):
    """
    L2 — Smart Money y estructura de volumen.
    Detecta si el volumen indica acumulación institucional:
    - Climactic Volume: spike de vol en velas alcistas
    - Volume Profile: dónde se concentra el volumen (soporte real)
    - OBV (On Balance Volume): acumulación neta
    - Volumen relativo creciente en tendencia (no pump & dump)
    Retorna score 0-25 + señales
    """
    if len(df) < 20:
        return 0, {}

    c = df["Close"]; v = df["Volume"]; o = df["Open"]
    score_sm = 0
    señales  = {}

    # ── OBV — On Balance Volume ──────────────────────────────
    # Si OBV sube mientras precio sube = acumulación real
    obv_delta = np.where(c.diff() > 0, v, np.where(c.diff() < 0, -v, 0))
    obv       = pd.Series(obv_delta).cumsum()
    # Pendiente OBV últimas 10 vs anteriores 10
    obv_vals  = obv.values
    if len(obv_vals) >= 20:
        obv_trend_rec  = np.mean(np.diff(obv_vals[-10:]))
        obv_trend_prev = np.mean(np.diff(obv_vals[-20:-10]))
        if obv_trend_rec > 0 and obv_trend_rec > obv_trend_prev:
            score_sm += 8; señales["OBV"] = ("🟢 ACUMULACIÓN", obv_trend_rec)
        elif obv_trend_rec > 0:
            score_sm += 4; señales["OBV"] = ("🟡 POSITIVO", obv_trend_rec)
        else:
            señales["OBV"] = ("🔴 DISTRIBUCIÓN", obv_trend_rec)

    # ── Volume Delta — velas alcistas vs bajistas ────────────
    # ¿El volumen es mayor en velas verdes que rojas?
    es_alcista  = c >= o
    vol_alcista = v[es_alcista].mean() if es_alcista.any() else 0
    vol_bajista = v[~es_alcista].mean() if (~es_alcista).any() else 1
    vol_delta_ratio = vol_alcista / (vol_bajista + 1e-10)

    if vol_delta_ratio > 1.4:
        score_sm += 8; señales["VOL_DELTA"] = ("🟢 COMPRADORES DOMINAN", round(vol_delta_ratio, 2))
    elif vol_delta_ratio > 1.1:
        score_sm += 4; señales["VOL_DELTA"] = ("🟡 LEVE DOMINIO COMPRA", round(vol_delta_ratio, 2))
    else:
        señales["VOL_DELTA"] = ("🔴 VENDEDORES DOMINAN", round(vol_delta_ratio, 2))

    # ── Volumen creciente en tendencia (no flat) ─────────────
    vol_sma_corto  = v.rolling(5).mean().iloc[-1]
    vol_sma_largo  = v.rolling(20).mean().iloc[-1]
    vol_tendencia  = vol_sma_corto / (vol_sma_largo + 1e-10)

    if vol_tendencia > 1.5:
        score_sm += 6; señales["VOL_TREND"] = ("🟢 ACELERANDO", round(vol_tendencia, 2))
    elif vol_tendencia > 1.2:
        score_sm += 3; señales["VOL_TREND"] = ("🟡 CRECIENDO", round(vol_tendencia, 2))
    else:
        señales["VOL_TREND"] = ("⚪ PLANO", round(vol_tendencia, 2))

    # ── Climactic volume — spike en ruptura ──────────────────
    vol_max_recent = v.iloc[-5:].max()
    vol_avg_base   = v.iloc[-25:-5].mean()
    climax_ratio   = vol_max_recent / (vol_avg_base + 1e-10)

    if climax_ratio > 3:
        score_sm += 3; señales["CLIMAX"] = ("🟢 CLIMAX ALCISTA", round(climax_ratio, 1))
    elif climax_ratio > 2:
        score_sm += 1; señales["CLIMAX"] = ("🟡 SPIKE VOL", round(climax_ratio, 1))
    else:
        señales["CLIMAX"] = ("⚪ SIN CLIMAX", round(climax_ratio, 1))

    return min(score_sm, 25), señales


def analizar_fuerza_relativa(df_sym, df_btc):
    """
    L4 — Fuerza Relativa vs BTC.
    Una altcoin que sube más que BTC en el mismo período
    indica capital rotando a ella (señal de fortaleza real).
    Retorna score 0-15 + ratio
    """
    if df_btc is None or df_btc.empty or df_sym is None or df_sym.empty:
        return 7, 1.0, "N/D"  # neutral

    try:
        ret_sym = (df_sym["Close"].iloc[-1] - df_sym["Close"].iloc[-10]) / (df_sym["Close"].iloc[-10] + 1e-10) * 100
        ret_btc = (df_btc["Close"].iloc[-1] - df_btc["Close"].iloc[-10]) / (df_btc["Close"].iloc[-10] + 1e-10) * 100
        rs      = ret_sym - ret_btc  # diferencial

        if rs > 8:
            return 15, rs, f"🟢 +{rs:.1f}% vs BTC (MUY FUERTE)"
        elif rs > 3:
            return 11, rs, f"🟢 +{rs:.1f}% vs BTC"
        elif rs > 0:
            return 7,  rs, f"🟡 +{rs:.1f}% vs BTC"
        elif rs > -5:
            return 4,  rs, f"🟡 {rs:.1f}% vs BTC"
        else:
            return 0,  rs, f"🔴 {rs:.1f}% vs BTC (DÉBIL)"
    except:
        return 7, 0, "N/D"


def analizar_momentum_tecnico(df):
    """
    L3 — Momentum técnico completo.
    Combina RSI, MACD, EMAs, StochRSI, Bollinger.
    Prioriza señales de inicio de tendencia, no de sobrecompra.
    Retorna score 0-20 + señales
    """
    if len(df) < 30:
        return 0, {}

    c = df["Close"]; score_m = 0; señales = {}

    # EMAs
    e9  = c.ewm(span=9,  adjust=False).mean()
    e21 = c.ewm(span=21, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean()
    e200= c.ewm(span=200,adjust=False).mean()

    # RSI
    d = c.diff()
    g = d.clip(lower=0).ewm(com=13,adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=13,adjust=False).mean()
    rsi = 100-(100/(1+g/(l+1e-10)))

    # MACD
    ef=c.ewm(span=12,adjust=False).mean(); es=c.ewm(span=26,adjust=False).mean()
    mac=ef-es; sig=mac.ewm(span=9,adjust=False).mean(); hist=mac-sig

    # Bollinger
    bb_mid=c.rolling(20).mean(); bb_std=c.rolling(20).std()
    bb_pct=(c-(bb_mid-2*bb_std))/(4*bb_std+1e-9)

    # Valores actuales
    rsi_v   = rsi.iloc[-1]; rsi_p  = rsi.iloc[-2]
    hist_v  = hist.iloc[-1]; hist_p = hist.iloc[-2]
    e9v=e9.iloc[-1]; e21v=e21.iloc[-1]; e50v=e50.iloc[-1]; e200v=e200.iloc[-1]
    c_now   = c.iloc[-1]; bb_v = bb_pct.iloc[-1]

    if any(pd.isna(x) for x in [rsi_v, hist_v, e9v]): return 0, {}

    # RSI en zona correcta (sano, no sobrecomprado)
    if 40 <= rsi_v <= 62 and rsi_v > rsi_p:
        score_m += 6; señales["RSI"] = (f"🟢 SANO {rsi_v:.0f}↑", 6)
    elif 35 <= rsi_v < 40:
        score_m += 4; señales["RSI"] = (f"🟡 SALIENDO SV {rsi_v:.0f}", 4)
    elif rsi_v > 70:
        señales["RSI"] = (f"🔴 SC {rsi_v:.0f}", 0)
    else:
        score_m += 2; señales["RSI"] = (f"⚪ {rsi_v:.0f}", 2)

    # MACD positivo y creciendo
    if hist_v > 0 and hist_v > hist_p:
        score_m += 6; señales["MACD"] = ("🟢 ACELERANDO", 6)
    elif hist_v > 0:
        score_m += 3; señales["MACD"] = ("🟡 POSITIVO", 3)
    elif hist_p <= 0 and hist_v > 0:
        score_m += 5; señales["MACD"] = ("🟢 CRUCE", 5)
    else:
        señales["MACD"] = ("🔴 NEGATIVO", 0)

    # Alineación de EMAs
    if e9v > e21v > e50v:
        score_m += 5; señales["EMAs"] = ("🟢 ALINEADAS", 5)
        if c_now > e200v:
            score_m += 3; señales["EMA200"] = ("🟢 SOBRE EMA200", 3)
    elif e9v > e21v:
        score_m += 2; señales["EMAs"] = ("🟡 PARCIAL", 2)
    else:
        señales["EMAs"] = ("🔴 DESALINEADAS", 0)

    return min(score_m, 20), señales


def analizar_kalman_trend(df):
    """
    L5 — Filtro Kalman para detectar tendencia real sostenida.
    Una tendencia Kalman que lleva varias velas al alza
    indica movimiento genuino, no ruido.
    Retorna score 0-10 + velocidad + aceleración
    """
    if len(df) < 20:
        return 5, 0, 0

    c  = df["Close"].values
    # Proxy Kalman rápido con EWM
    pk = pd.Series(c).ewm(span=6, adjust=False).mean().values
    vel= np.diff(pk)
    acc= np.diff(vel)

    vel_actual = vel[-1]   if len(vel)>0   else 0
    vel_media  = np.mean(vel[-5:])  if len(vel)>=5  else vel_actual
    acc_actual = acc[-1]   if len(acc)>0   else 0

    # Velas consecutivas con Kalman alcista
    consec_al = 0
    for v2 in reversed(vel[-8:]):
        if v2 > 0: consec_al += 1
        else: break

    precio = c[-1]
    vel_pct = vel_actual / (precio + 1e-10) * 100

    if consec_al >= 5 and vel_media > 0 and acc_actual >= 0:
        return 10, vel_pct, acc_actual
    elif consec_al >= 3 and vel_media > 0:
        return 7, vel_pct, acc_actual
    elif consec_al >= 2:
        return 4, vel_pct, acc_actual
    elif vel_actual < 0:
        return 0, vel_pct, acc_actual
    else:
        return 2, vel_pct, acc_actual


@st.cache_data(ttl=300)
def get_btc_data_scanner():
    """BTC como referencia para fuerza relativa — cacheado 5 min."""
    try:
        df_b = binance_descargar("BTCUSDT", "1d", 30)
        return df_b if not df_b.empty else None
    except:
        return None


def analizar_subida_sostenida(df, sym, df_btc=None):
    """
    Motor principal: 5 capas de análisis.
    Solo pasa el filtro si MÚLTIPLES capas confirman.
    Diseñado para subidas de días/semanas, no horas.
    """
    if df is None or df.empty or len(df) < 35:
        return None
    try:
        # ── Ejecutar las 5 capas ─────────────────────────────
        score_w, desc_w   = detectar_wyckoff_acumulacion(df)        # 0-30
        score_sm, sel_sm  = detectar_smart_money_volumen(df)        # 0-25
        score_mt, sel_mt  = analizar_momentum_tecnico(df)           # 0-20
        score_fr, rs_val, desc_fr = analizar_fuerza_relativa(df, df_btc)  # 0-15
        score_kl, vel_pct, acc    = analizar_kalman_trend(df)       # 0-10

        total_max   = 30 + 25 + 20 + 15 + 10  # = 100
        score_raw   = score_w + score_sm + score_mt + score_fr + score_kl
        score_final = int(round(score_raw / total_max * 100))

        # ── Filtro de calidad — Wyckoff + Smart Money obligatorios ──
        # Una subida sin base de acumulación NO es sostenible
        if score_w < 8:    score_final = min(score_final, 48)
        if score_sm < 8:   score_final = min(score_final, 52)

        # ── Penalización por sobrecompra (ya subió demasiado) ──
        c = df["Close"]
        ret_10 = (c.iloc[-1] - c.iloc[-11]) / (c.iloc[-11] + 1e-10) * 100 if len(c) > 11 else 0
        ret_3  = (c.iloc[-1] - c.iloc[-4])  / (c.iloc[-4]  + 1e-10) * 100 if len(c) > 4  else 0

        timing_ok = True
        if ret_10 > 40:
            score_final = min(score_final, 45)
            timing_ok = False
        elif ret_10 > 25:
            score_final = max(0, score_final - 15)

        # ── Clasificación de calidad ─────────────────────────
        capas_fuertes = sum([
            score_w  >= 18,   # Wyckoff fuerte
            score_sm >= 15,   # Smart money fuerte
            score_mt >= 13,   # Momentum fuerte
            score_fr >= 10,   # FR fuerte
            score_kl >= 7,    # Kalman fuerte
        ])

        if score_final >= 75 and capas_fuertes >= 4:
            calidad = "💎 SUBIDA EXCELENTE"
            calidad_col = "#00ff88"
            duracion_est = "Semanas"
        elif score_final >= 62 and capas_fuertes >= 3:
            calidad = "🚀 SUBIDA SÓLIDA"
            calidad_col = "#44ffaa"
            duracion_est = "Días-Semanas"
        elif score_final >= 50 and capas_fuertes >= 2:
            calidad = "⚡ SUBIDA PROBABLE"
            calidad_col = "#ffaa00"
            duracion_est = "Días"
        elif score_final >= 38:
            calidad = "👀 EN FORMACIÓN"
            calidad_col = "#4488ff"
            duracion_est = "Incierto"
        else:
            return None  # No clasificado

        # ── Niveles de entrada ───────────────────────────────
        precio = c.iloc[-1]
        h = df["High"]; l = df["Low"]
        tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
        atr = tr.ewm(span=14,adjust=False).mean().iloc[-1]
        if pd.isna(atr) or atr == 0: atr = precio * 0.02

        entry = precio
        sl    = precio - 1.5*atr
        tp1   = precio + 2.5*atr
        tp2   = precio + 5.0*atr
        tp3   = precio + 8.0*atr   # objetivo extendido para subidas de semanas
        rr    = (tp1-entry)/(entry-sl+1e-10)

        e21_v = c.ewm(span=21,adjust=False).mean().iloc[-1]
        e50_v = c.ewm(span=50,adjust=False).mean().iloc[-1]

        return {
            "symbol":      sym,
            "score":       score_final,
            "calidad":     calidad,
            "calidad_col": calidad_col,
            "duracion_est":duracion_est,
            "capas":       capas_fuertes,
            # Scores por capa
            "sc_wyckoff":  score_w,
            "sc_smart":    score_sm,
            "sc_momentum": score_mt,
            "sc_fr":       score_fr,
            "sc_kalman":   score_kl,
            # Descripciones
            "desc_wyckoff":desc_w,
            "desc_fr":     desc_fr,
            "sel_smart":   sel_sm,
            "sel_momentum":sel_mt,
            # Precio y niveles
            "precio":      precio,
            "ret_3":       round(ret_3,  2),
            "ret_10":      round(ret_10, 2),
            "rs_btc":      round(rs_val, 2),
            "vel_kalman":  round(vel_pct,4),
            "entry":       entry,
            "sl":          sl,
            "tp1":         tp1,
            "tp2":         tp2,
            "tp3":         tp3,
            "rr":          round(rr, 2),
            "atr_pct":     round(atr/precio*100, 2),
            "e21":         e21_v,
            "e50":         e50_v,
            "timing_ok":   timing_ok,
        }
    except:
        return None


def ejecutar_scanner_sostenido(pares, interval_bn, dias, progreso_bar):
    """Ejecuta el scanner con barra de progreso."""
    df_btc = get_btc_data_scanner()
    resultados = []
    total = len(pares)
    for i, sym in enumerate(pares):
        progreso_bar.progress((i+1)/total, text=f"🔭 {sym} ({i+1}/{total})")
        try:
            df_s = binance_descargar(sym, interval_bn, dias)
            res  = analizar_subida_sostenida(df_s, sym, df_btc)
            if res: resultados.append(res)
        except:
            pass
        time.sleep(0.06)
    resultados.sort(key=lambda x: x["score"], reverse=True)
    return resultados


# ══════════════════════════════════════════════════════════════
#  UI — SCANNER DE SUBIDAS SOSTENIDAS
# ══════════════════════════════════════════════════════════════
st.sidebar.divider()
st.sidebar.markdown("**🏆 SCANNER SOSTENIDO**")

with st.sidebar:
    tf_scan_sos = st.selectbox("⏱ TF Scanner:",
                               ["1D · 30 días","1D · 90 días","4H · 10 días","1H · 3 días"],
                               key="tf_scan_sos",
                               index=0)
    min_sc_sos = st.slider("Score mínimo:", 35, 80, 50, key="sc_sos")
    run_scanner_sos = st.button("🏆 BUSCAR SUBIDAS SOSTENIDAS", use_container_width=True)

if run_scanner_sos:
    st.divider()
    st.markdown("## 🏆 SCANNER DE SUBIDAS SOSTENIDAS")
    st.markdown("""
    <div style="background:#080c18;border:1px solid #1a2a4a;border-radius:8px;
                padding:12px 16px;font-family:'Share Tech Mono',monospace;
                font-size:0.78rem;color:#7aabff;margin-bottom:10px">
        <b style="color:#4488ff;font-size:0.85rem">🏆 Modelo de 5 Capas (Wyckoff + Smart Money)</b><br><br>
        <b style="color:#00ff88">L1 Wyckoff</b> → Acumulación previa sólida (base de semanas)<br>
        <b style="color:#44aaff">L2 Smart Money</b> → OBV, Volume Delta, compradores dominantes<br>
        <b style="color:#ffaa00">L3 Momentum</b> → RSI sano + MACD + EMAs alineadas<br>
        <b style="color:#ff88ff">L4 Fuerza Relativa</b> → Más fuerte que BTC (capital rotando)<br>
        <b style="color:#88ffdd">L5 Kalman</b> → Tendencia real sostenida, no ruido<br><br>
        <b style="color:#ffdd44">Solo aparecen criptos donde ≥3 capas confirman la subida.</b>
    </div>
    """, unsafe_allow_html=True)

    tf_map_sos = {
        "1D · 30 días":  ("1d", 30),
        "1D · 90 días":  ("1d", 90),
        "4H · 10 días":  ("4h", 10),
        "1H · 3 días":   ("1h", 3),
    }
    iv_sos, d_sos = tf_map_sos[tf_scan_sos]

    prog_sos = st.progress(0, text="Iniciando...")
    t0 = time.time()
    resultados_sos = ejecutar_scanner_sostenido(SCAN_PARES_SOSTENIDO, iv_sos, d_sos, prog_sos)
    prog_sos.empty()
    t_total = time.time() - t0

    filtrados = [r for r in resultados_sos if r["score"] >= min_sc_sos]
    st.caption(f"Escaneados: {len(SCAN_PARES_SOSTENIDO)} · Tiempo: {t_total:.0f}s · Calificados: {len(filtrados)}")

    if not filtrados:
        st.info("""
        🔍 **No se encontraron subidas sostenidas ahora.**

        Esto es **normal y es buena señal** — el scanner es exigente por diseño.
        Cuando el mercado está en acumulación o distribución, no hay señales válidas.

        **Prueba:**
        - Timeframe `1D · 90 días` (más contexto = mejor análisis Wyckoff)
        - Baja el score mínimo a 40
        - Vuelve en 24-48 horas (los ciclos cambian)
        """)
    else:
        # ── OVERVIEW — mapa de calor de scores ──────────────
        fig_ov = plt.figure(figsize=(14, 5), facecolor=BG)
        ax_ov  = fig_ov.add_subplot(1,1,1); estilizar_ax(ax_ov)
        top_vis = filtrados[:18]
        syms_v  = [r["symbol"].replace("USDT","") for r in top_vis]
        scrs_v  = [r["score"] for r in top_vis]
        bars_v  = []
        for r in top_vis:
            if r["score"] >= 75:   bars_v.append("#00ff88")
            elif r["score"] >= 62: bars_v.append("#44ffaa")
            elif r["score"] >= 50: bars_v.append("#ffaa00")
            else:                  bars_v.append("#4488ff")

        brs = ax_ov.bar(syms_v, scrs_v, color=bars_v, alpha=0.85, width=0.65, zorder=3)
        for bar, r in zip(brs, top_vis):
            ax_ov.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                      f"{r['score']}", ha="center", va="bottom", color="white", fontsize=8, fontweight="bold")
            ax_ov.text(bar.get_x()+bar.get_width()/2, bar.get_height()//2,
                      r["calidad"].split()[0], ha="center", va="center", color="white", fontsize=9)

        # Líneas de referencia
        ax_ov.axhline(75, color="#00ff88", lw=1.2, ls="--", alpha=0.7, label="💎 Excelente")
        ax_ov.axhline(62, color="#ffaa00", lw=1,   ls="--", alpha=0.5, label="🚀 Sólida")
        ax_ov.axhline(50, color="#4488ff", lw=0.8, ls=":",  alpha=0.4, label="⚡ Probable")
        ax_ov.set_ylim(0, 110)
        ax_ov.set_title(f"Calidad de Subida Sostenida — 5 Capas · {tf_scan_sos}", color="#4488ff", fontsize=10)
        ax_ov.legend(fontsize=7, framealpha=0.3, loc="upper right")
        ax_ov.grid(axis="y", color="#0d1a2e", lw=0.5, alpha=0.5)
        ax_ov.tick_params(axis="x", labelrotation=35, labelsize=8, colors="#4a6080")
        plt.tight_layout(); render_fig(fig_ov)

        # ── RADAR de las 5 capas (Top 5) ────────────────────
        top5 = filtrados[:5]
        if len(top5) >= 2:
            st.markdown("### 📡 Radar de capas — Top oportunidades")
            fig_rad = plt.figure(figsize=(14, 4), facecolor=BG)
            n_cats  = 5
            cats    = ["Wyckoff\n/30", "Smart\nMoney/25", "Momentum\n/20", "F.Relativa\n/15", "Kalman\n/10"]
            maxs    = [30, 25, 20, 15, 10]

            for idx_r, r in enumerate(top5):
                ax_r = fig_rad.add_subplot(1, len(top5), idx_r+1); estilizar_ax(ax_r)
                vals = [r["sc_wyckoff"], r["sc_smart"], r["sc_momentum"], r["sc_fr"], r["sc_kalman"]]
                pcts = [v/m for v,m in zip(vals,maxs)]
                x    = np.arange(n_cats)
                ax_r.bar(x, [v*100 for v in pcts], color=r["calidad_col"], alpha=0.7, width=0.6)
                ax_r.set_xticks(x); ax_r.set_xticklabels(cats, fontsize=6, color="#4a6080")
                ax_r.set_ylim(0, 110); ax_r.set_yticks([])
                ax_r.set_title(f"{r['symbol'].replace('USDT','')}\n{r['score']}/100",
                              color=r["calidad_col"], fontsize=8, fontweight="bold")
                for xi, (pct, val) in enumerate(zip(pcts, vals)):
                    ax_r.text(xi, pct*100+2, f"{val}", ha="center", fontsize=7, color="white")
                ax_r.axhline(70, color="#2a4060", lw=0.5, ls="--")

            plt.tight_layout(); render_fig(fig_rad)

        # ── TABS por calidad ─────────────────────────────────
        excelentes = [r for r in filtrados if "EXCELENTE" in r["calidad"]]
        solidas    = [r for r in filtrados if "SÓLIDA"    in r["calidad"]]
        probables  = [r for r in filtrados if "PROBABLE"  in r["calidad"]]
        formacion  = [r for r in filtrados if "FORMACIÓN" in r["calidad"]]

        t_labels = []
        grupos_t = []
        if excelentes: t_labels.append(f"💎 Excelente ({len(excelentes)})"); grupos_t.append(excelentes)
        if solidas:    t_labels.append(f"🚀 Sólida ({len(solidas)})");    grupos_t.append(solidas)
        if probables:  t_labels.append(f"⚡ Probable ({len(probables)})"); grupos_t.append(probables)
        if formacion:  t_labels.append(f"👀 Formación ({len(formacion)})");grupos_t.append(formacion)
        if not t_labels: t_labels=["📋 Todos"]; grupos_t=[filtrados]

        tabs_sos = st.tabs(t_labels)

        for tab_obj, grupo in zip(tabs_sos, grupos_t):
            with tab_obj:
                # Cards top 3
                top3 = grupo[:3]
                cols3 = st.columns(min(3, len(top3)))
                for i, r in enumerate(top3):
                    pf = r["precio"]
                    pfmt = f"${pf:,.6f}" if pf<1 else f"${pf:,.4f}" if pf<10 else f"${pf:,.2f}"
                    rr_col = "#00ff88" if r["rr"]>=2.5 else ("#ffaa00" if r["rr"]>=1.5 else "#ff3355")

                    # Badges de capas
                    capas_html = ""
                    capa_items = [
                        ("L1", r["sc_wyckoff"],  30, r["calidad_col"]),
                        ("L2", r["sc_smart"],    25, "#44aaff"),
                        ("L3", r["sc_momentum"], 20, "#ffaa00"),
                        ("L4", r["sc_fr"],       15, "#ff88ff"),
                        ("L5", r["sc_kalman"],   10, "#88ffdd"),
                    ]
                    for lbl, sc_c, mx_c, col_c in capa_items:
                        pct_c = sc_c/mx_c*100
                        ic    = "✅" if pct_c>=60 else ("⚠️" if pct_c>=35 else "❌")
                        capas_html += f'<span style="color:{col_c};font-size:0.7rem">{lbl}:{sc_c}{ic} </span>'

                    cols3[i].markdown(f"""
                    <div style="background:#08090f;border:1.5px solid {r['calidad_col']};
                                border-radius:10px;padding:14px;font-family:'Share Tech Mono',monospace">
                        <div style="font-family:'Orbitron',monospace;font-size:1rem;
                                    color:{r['calidad_col']};font-weight:700">
                            {r['symbol'].replace('USDT','/USDT')}
                        </div>
                        <div style="font-size:0.68rem;color:#4488ff;margin:2px 0">{r['calidad']}</div>
                        <div style="font-size:1.6rem;font-weight:700;color:{r['calidad_col']}">{r['score']}/100</div>
                        <div style="color:#c0d8ff;font-size:0.9rem;margin:4px 0">
                            {pfmt}
                            <span style="color:{'#00ff88' if r['ret_3']>0 else '#ff3355'}">{r['ret_3']:+.1f}%/3v</span>
                        </div>
                        <div style="margin:5px 0">{capas_html}</div>
                        <div style="font-size:0.7rem;color:#2a4060;margin:4px 0">
                            FR vs BTC: <b style="color:{'#00ff88' if r['rs_btc']>0 else '#ff3355'}">{r['rs_btc']:+.1f}%</b>
                            &nbsp;·&nbsp;Duración est.: <b style="color:{r['calidad_col']}">{r['duracion_est']}</b>
                        </div>
                        <div style="border-top:1px solid #0d1a2e;padding-top:7px;margin-top:6px;font-size:0.7rem;color:#2a4060">
                            Entry <span style="color:#00ff88">{pfmt}</span><br>
                            SL <span style="color:#ff3355">{fp(r['sl'])}</span> &nbsp;
                            TP1 <span style="color:#ffaa00">{fp(r['tp1'])}</span> &nbsp;
                            TP2 <span style="color:#ffdd44">{fp(r['tp2'])}</span><br>
                            TP3 <span style="color:#ffffff">{fp(r['tp3'])}</span> (objetivo semanas)<br>
                            R/R <span style="color:{rr_col}">×{r['rr']}</span> &nbsp;·&nbsp; ATR {r['atr_pct']}%
                        </div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("&nbsp;")

                # Tabla completa con column_config
                filas = []
                for r in grupo:
                    pf = r["precio"]
                    pfmt = f"${pf:,.6f}" if pf<1 else f"${pf:,.4f}" if pf<10 else f"${pf:,.2f}"
                    filas.append({
                        "Par":        r["symbol"].replace("USDT","/USDT"),
                        "Score":      r["score"],
                        "Calidad":    r["calidad"],
                        "Precio":     pfmt,
                        "Ret 3v":    f"{r['ret_3']:+.1f}%",
                        "Ret 10v":   f"{r['ret_10']:+.1f}%",
                        "FR vs BTC": f"{r['rs_btc']:+.1f}%",
                        "L1 Wyckoff":r["sc_wyckoff"],
                        "L2 SmartM": r["sc_smart"],
                        "L3 Moment": r["sc_momentum"],
                        "L4 F.Rel":  r["sc_fr"],
                        "L5 Kalman": r["sc_kalman"],
                        "Capas OK":  r["capas"],
                        "R/R":        r["rr"],
                        "Duración":  r["duracion_est"],
                    })

                if filas:
                    df_tab = pd.DataFrame(filas)
                    st.dataframe(df_tab, use_container_width=True, hide_index=True,
                                column_config={
                                    "Score":       st.column_config.ProgressColumn("Score",     min_value=0, max_value=100, format="%d"),
                                    "L1 Wyckoff":  st.column_config.ProgressColumn("L1 Wyckoff",min_value=0, max_value=30,  format="%d"),
                                    "L2 SmartM":   st.column_config.ProgressColumn("L2 Smart",  min_value=0, max_value=25,  format="%d"),
                                    "L3 Moment":   st.column_config.ProgressColumn("L3 Moment", min_value=0, max_value=20,  format="%d"),
                                    "L4 F.Rel":    st.column_config.ProgressColumn("L4 F.Rel",  min_value=0, max_value=15,  format="%d"),
                                    "L5 Kalman":   st.column_config.ProgressColumn("L5 Kalman", min_value=0, max_value=10,  format="%d"),
                                    "R/R":         st.column_config.NumberColumn("R/R", format="%.1f"),
                                })

        # ── Diagnóstico detallado del #1 ─────────────────────
        if filtrados:
            st.divider()
            top1 = filtrados[0]
            st.markdown(f"### 🔬 Diagnóstico completo — {top1['symbol']}")

            col_d1, col_d2 = st.columns([1, 2])
            with col_d1:
                pf = top1["precio"]
                pfmt = f"${pf:,.6f}" if pf<1 else f"${pf:,.4f}" if pf<10 else f"${pf:,.2f}"
                st.markdown(f"""
                <div class="mc" style="border-left:3px solid {top1['calidad_col']}">
                    <div class="ml">Calificación</div>
                    <div class="mv" style="color:{top1['calidad_col']}">{top1['calidad']}</div>
                    <br>
                    <div class="ml">Score Total</div>
                    <div style="font-size:2rem;font-weight:700;color:{top1['calidad_col']}">{top1['score']}/100</div>
                    <br>
                    <div class="ml">Precio · Duración estimada</div>
                    <div style="color:#c0d8ff">{pfmt} · <b>{top1['duracion_est']}</b></div>
                    <br>
                    <div class="ml">Fuerza vs BTC</div>
                    <div style="color:{'#00ff88' if top1['rs_btc']>0 else '#ff3355'};font-size:1rem">{top1['rs_btc']:+.1f}%</div>
                    <br>
                    <div class="ml">Capas confirmadas</div>
                    <div style="color:#ffaa00;font-size:1.1rem">{top1['capas']}/5 capas</div>
                </div>""", unsafe_allow_html=True)

            with col_d2:
                # Barra de progreso de cada capa
                capas_detalle = [
                    ("L1 — Wyckoff / Acumulación",  top1["sc_wyckoff"],  30, "#00ff88",  top1["desc_wyckoff"]),
                    ("L2 — Smart Money / Volumen",  top1["sc_smart"],    25, "#44aaff",  str(top1["sel_smart"])),
                    ("L3 — Momentum Técnico",        top1["sc_momentum"], 20, "#ffaa00",  str(top1["sel_momentum"])),
                    ("L4 — Fuerza Relativa vs BTC",  top1["sc_fr"],       15, "#ff88ff",  top1["desc_fr"]),
                    ("L5 — Kalman Trend",             top1["sc_kalman"],   10, "#88ffdd",  f"Vel: {top1['vel_kalman']:+.4f}%/vela"),
                ]
                for nombre, sc_c, mx_c, col_c, det in capas_detalle:
                    pct_c = sc_c/mx_c*100
                    ic    = "✅" if pct_c>=60 else ("⚠️" if pct_c>=35 else "❌")
                    st.markdown(f"""
                    <div style="margin:6px 0;font-family:'Share Tech Mono',monospace">
                        <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#4a6080">
                            <span style="color:{col_c}">{ic} {nombre}</span>
                            <span style="color:{col_c};font-weight:700">{sc_c}/{mx_c} ({pct_c:.0f}%)</span>
                        </div>
                        <div class="bw"><div class="bf" style="width:{pct_c:.0f}%;background:{col_c}"></div></div>
                        <div style="font-size:0.65rem;color:#2a4060;margin-top:1px">{str(det)[:80]}</div>
                    </div>""", unsafe_allow_html=True)

        # ── Nota educativa ────────────────────────────────────
        st.markdown("""
        <div style="background:#0d1428;border-left:3px solid #4488ff;border-radius:6px;
                    padding:12px 16px;margin-top:12px;font-family:'Share Tech Mono',monospace;
                    font-size:0.77rem;color:#7aabff">
            <b style="color:#4488ff">📚 CÓMO OPERAR ESTOS RESULTADOS:</b><br>
            · <b>💎 Excelente</b> — Entrada directa con 3-5% capital, SL ajustado, TP3 es el objetivo real<br>
            · <b>🚀 Sólida</b> — Entrada en pullback a EMA21, reducir tamaño si no hay retroceso<br>
            · <b>⚡ Probable</b> — Esperar confirmación de vela adicional antes de entrar<br>
            · <b>Regla de oro:</b> Si L1 (Wyckoff) y L2 (Smart Money) no están verdes, NO entres<br>
            · <b>Trailing:</b> Mueve SL a breakeven al llegar a TP1. Al TP2 → cierra 50%. Deja correr al TP3.
        </div>
        """, unsafe_allow_html=True)
