# ============================================================
# AI.LINO QUANTUM ENGINE v4.0
# Filtro Tendencia - Entradas Precisas - Trailing - MTF
# Backtesting - Confluence Score - HMM 5D
# ============================================================
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

# ==============================================================
#  TIMEFRAMES
# ==============================================================
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

# ==============================================================
#  FUENTES DE DATOS
# ==============================================================
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

# ==============================================================
#  INDICADORES TCNICOS
# ==============================================================
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

# ==============================================================
#  FILTRO DE TENDENCIA (mejora #1 de Grok)
# ==============================================================
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

    # Pendiente EMA50 (ltimas 5 velas)
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

# ==============================================================
#  CONFIRMACIN MULTI-TIMEFRAME (MTF)  mejora #4 de Grok
# ==============================================================
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

# ==============================================================
#  HMM MEJORADO  5 features, mltiples semillas
# ==============================================================
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

# ==============================================================
#  CONFLUENCE SCORE (mejora #2 de Grok  ms exigente)
# ==============================================================
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

    # RSI  20pts (ms sensible en extremos)
    if rsi<22:    p.append(("RSI Extremo S.venta",20,20,"🟢"))
    elif rsi<32:  p.append(("RSI Sobreventa",16,20,"🟢"))
    elif rsi<45:  p.append(("RSI Zona baja",11,20,"🟡"))
    elif rsi<55:  p.append(("RSI Neutral",7,20,"⚪"))
    elif rsi<65:  p.append(("RSI Zona alta",4,20,"🟡"))
    elif rsi<76:  p.append(("RSI Sobrecompra",2,20,"🔴"))
    else:         p.append(("RSI Extremo SC",0,20,"🔴"))

    # MACD  18pts
    if prev<=0 and hist>0:          p.append(("MACD Cruce alcista",18,18,"🟢"))
    elif hist>0 and hist>prev*1.15: p.append(("MACD Acelerando ↑",14,18,"🟢"))
    elif hist>0:                    p.append(("MACD Positivo",9,18,"🟡"))
    elif prev>=0 and hist<0:        p.append(("MACD Cruce bajista",0,18,"🔴"))
    elif hist<prev*1.15:            p.append(("MACD Cayendo ↓",3,18,"🔴"))
    else:                           p.append(("MACD Negativo",5,18,"🔴"))

    # Bollinger %B  12pts
    if pct<-0.05:  p.append(("BB Bajo banda inf",12,12,"🟢"))
    elif pct<0.2:  p.append(("BB Zona baja",9,12,"🟡"))
    elif pct<0.55: p.append(("BB Centro",6,12,"⚪"))
    elif pct<0.85: p.append(("BB Zona alta",3,12,"🟡"))
    else:          p.append(("BB Sobre banda sup",0,12,"🔴"))

    # EMAs  18pts (alineacin completa)
    es = 0
    if not any(pd.isna(v) for v in [e9,e21,e50]):
        close_v = ind["bb_mid"].iloc[-1]
        if e9>e21>e50 and close_v>e9:   es=18
        elif e9>e21>e50:                 es=14
        elif e9>e21:                     es=9
        elif e9>e50:                     es=5
        else:                            es=1
    p.append(("EMAs alineadas",es,18,"🟢" if es>=14 else("🟡" if es>=7 else "🔴")))

    # Volumen  12pts (contexto direccional)
    if vr>2.5 and chg1<-1.5:  p.append(("Vol+caída=capitulación",11,12,"🟢"))
    elif vr>2.5 and chg1>1:   p.append(("Vol+subida=impulso",12,12,"🟢"))
    elif vr>1.5:               p.append(("Vol elevado",8,12,"🟡"))
    elif vr>0.8:               p.append(("Vol normal",6,12,"⚪"))
    else:                      p.append(("Vol muy seco",1,12,"🔴"))

    # StochRSI  10pts
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

    #  Ajuste por HMM urgencia 
    hmm_adj = {1:+8, 2:+3, 3:-8, 4:-18}.get(hmm_urgencia, 0)

    #  Ajuste por tendencia 
    tend_adj = 0
    tf = tendencia_info["fuerza"]
    if tf >= 80:   tend_adj = +10
    elif tf >= 65: tend_adj = +5
    elif tf <= 20: tend_adj = -15
    elif tf <= 35: tend_adj = -8

    #  Ajuste MTF 
    mtf_adj = 0
    if mtf_info["ok"]:
        if mtf_info["tendencia"]=="ALCISTA" and mtf_info["score_sup"]>60: mtf_adj=+8
        elif mtf_info["tendencia"]=="BAJISTA" and mtf_info["score_sup"]<40: mtf_adj=-10

    #  Bonus de confluencia mxima (Grok) 
    bonus = 0
    if pct<0.15 and rsi<35 and hmm_urgencia<=2 and vr>1.5:
        bonus = +15  # Confluencia perfecta de compra

    score_final = np.clip(score_base + hmm_adj + tend_adj + mtf_adj + bonus, 0, 100)
    return int(round(score_final)), p, {"hmm":hmm_adj,"tend":tend_adj,"mtf":mtf_adj,"bonus":bonus}

# ==============================================================
#  ENTRADAS PRECISAS + TRAILING STOP (mejora #3 de Grok)
# ==============================================================
def generar_señal_precisa(score, ind, tendencia_info, mtf_info, fuente):
    precio = ind["bb_mid"].iloc[-1]
    atr    = ind["atr"].iloc[-1]
    if pd.isna(atr) or atr==0: atr = precio*0.015
    bb_lo  = ind["bb_lo"].iloc[-1]
    bb_up  = ind["bb_up"].iloc[-1]
    is_crypto = fuente in ("binance","coingecko")
    mult   = 1.0 if is_crypto else 1.3   # cripto ATR ya es grande

    # Bloquear seales contra tendencia fuerte
    if score >= 68 and tendencia_info["bloquear_long"]:
        score = min(score, 54)  # degradar si tendencia bajista fuerte
    if score <= 32 and tendencia_info["bloquear_short"]:
        score = max(score, 40)

    if score >= 80:
        # LONG AGRESIVO  tendencia fuerte + confluencia alta
        entry  = precio
        sl     = max(precio - 1.3*mult*atr, bb_lo*0.995)
        tp1    = precio + 2.2*mult*atr
        tp2    = precio + 4.0*mult*atr
        trail  = 1.5*mult*atr
        return ("🟢 LONG AGRESIVO","sc", entry,sl,tp1,tp2,trail,
                f"Confluencia máxima. Entrada directa en {fp(entry)}. "
                f"Trailing: {fp(trail)} por vela.")

    elif score >= 68:
        # LONG CONSERVADOR  esperar pullback a zona BB inferior
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

# ==============================================================
#  BACKTESTING VECTORIZADO (mejora #5 de Grok)
# ==============================================================
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
            # Seal de entrada: cruce MACD alcista + RSI < 65 + EMA9>EMA21
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
            # Trailing stop dinmico
            nuevo_sl = precio - trail
            sl_actual = max(sl_actual, nuevo_sl)   # solo sube

            # Salida por SL o seal bajista
            cruce_baj = (macd[i]<0 and macd[i-1]>=0)
            if precio <= sl_actual or cruce_baj:
                pnl    = (precio - entrada) / entrada * 100
                capital *= (1 + pnl/100)
                trades.append({"tipo":"EXIT","precio":precio,"idx":i,"pnl_pct":pnl})
                posicion = 0

        equity.append(capital)

    # Cerrar posicin abierta al final
    if posicion == 1 and len(c)>0:
        pnl = (c[-1]-entrada)/entrada*100
        capital *= (1+pnl/100)
        trades.append({"tipo":"EXIT","precio":c[-1],"idx":n-1,"pnl_pct":pnl})
        equity[-1] = capital

    # Mtricas
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

# ==============================================================
#  MDULOS CUNTICOS
# ==============================================================
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

# ==============================================================
#  UTILIDADES DE RENDERIZADO
# ==============================================================
def estilizar_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors="#2a4060", labelsize=7)
    for sp in ax.spines.values(): sp.set_color("#0d1a2e")

def render_fig(fig):
    st.pyplot(fig)
    plt.close(fig)

# ==============================================================

# ============================================================
# AI.LINO APEX - UI LIMPIA
# Sidebar: buscador + agregar posicion con monto
# Main: APEX scanner + seguimiento en tiempo real
# ============================================================

import os

# Supabase
try:
    from supabase import create_client
    _su = os.environ.get("SUPABASE_URL","")
    _sk = os.environ.get("SUPABASE_KEY","")
    supa    = create_client(_su, _sk) if _su and _sk else None
    SUPA_OK = bool(supa)
except:
    supa = None; SUPA_OK = False

CG = "https://api.coingecko.com/api/v3"

# Pares disponibles con CoinGecko ID
PARES = [
    ("bitcoin","BTC"),("ethereum","ETH"),("solana","SOL"),
    ("ripple","XRP"),("dogecoin","DOGE"),("cardano","ADA"),
    ("avalanche-2","AVAX"),("chainlink","LINK"),("near","NEAR"),
    ("injective-protocol","INJ"),("aptos","APT"),("arbitrum","ARB"),
    ("optimism","OP"),("sui","SUI"),("uniswap","UNI"),
    ("aave","AAVE"),("fetch-ai","FET"),("matic-network","MATIC"),
    ("shiba-inu","SHIB"),("pepe","PEPE"),("litecoin","LTC"),
    ("bitcoin-cash","BCH"),("maker","MKR"),("the-sandbox","SAND"),
]
SYM_TO_CID = {sym:cid for cid,sym in PARES}
CID_TO_SYM = {cid:sym for cid,sym in PARES}

APEX_SCAN_PARES = [
    ("bitcoin","BTC"),("ethereum","ETH"),("solana","SOL"),
    ("avalanche-2","AVAX"),("near","NEAR"),("injective-protocol","INJ"),
    ("aptos","APT"),("arbitrum","ARB"),("optimism","OP"),("sui","SUI"),
    ("fetch-ai","FET"),("chainlink","LINK"),("uniswap","UNI"),
    ("aave","AAVE"),("ripple","XRP"),("dogecoin","DOGE"),
    ("matic-network","MATIC"),("near","NEAR"),
]
seen_ap=set()
APEX_SCAN_PARES=[p for p in APEX_SCAN_PARES if not(p[1] in seen_ap or seen_ap.add(p[1]))]


# ============================================================
# SUPABASE - posiciones
# ============================================================
def db_guardar(pos):
    if SUPA_OK:
        try:
            supa.table("ailino_posicion_activa") \
                .update({"activa":False}) \
                .eq("sym",pos["sym"]).eq("activa",True).execute()
            supa.table("ailino_posicion_activa").insert({
                "sym":pos["sym"],"cid":pos["cid"],
                "entry":float(pos["entry"]),"sl":float(pos["sl"]),
                "tp1":float(pos["tp1"]),"tp2":float(pos["tp2"]),
                "tp3":float(pos["tp3"]),"trail":float(pos["trail"]),
                "trail_sl":float(pos["sl"]),"max_precio":float(pos["entry"]),
                "atr_pct":float(pos.get("atr_pct",2)),
                "score":int(pos.get("score",0)),
                "monto":float(pos.get("monto",0)),
                "tiempo":datetime.utcnow().isoformat(),
                "activa":True,"estado":"SEGUIMIENTO",
            }).execute()
        except: st.session_state["_pos"]=pos
    else: st.session_state["_pos"]=pos

def db_cargar():
    if SUPA_OK:
        try:
            r=supa.table("ailino_posicion_activa") \
               .select("*").eq("activa",True) \
               .order("created_at",desc=True).limit(1).execute()
            if r.data: return r.data[0]
        except: pass
    return st.session_state.get("_pos")

def db_cerrar(sym,pnl):
    if SUPA_OK:
        try:
            supa.table("ailino_posicion_activa") \
                .update({"activa":False,"estado":f"CERRADA {pnl:+.2f}%"}) \
                .eq("sym",sym).eq("activa",True).execute()
        except: pass
    st.session_state.pop("_pos",None)

def db_update_trail(sym,trail_sl,max_p):
    if SUPA_OK:
        try:
            supa.table("ailino_posicion_activa") \
                .update({"trail_sl":float(trail_sl),"max_precio":float(max_p)}) \
                .eq("sym",sym).eq("activa",True).execute()
        except: pass
    p=st.session_state.get("_pos")
    if p: p["trail_sl"]=trail_sl; p["max_precio"]=max_p; st.session_state["_pos"]=p


# ============================================================
# DATOS EN TIEMPO REAL
# ============================================================
@st.cache_data(ttl=25)
def get_precio(cid):
    try:
        r=requests.get(f"{CG}/simple/price",params={
            "ids":cid,"vs_currencies":"usd",
            "include_1hr_change":"true","include_24hr_change":"true",
        },timeout=8)
        if r.status_code==200:
            d=r.json().get(cid,{})
            return {"p":float(d.get("usd",0) or 0),
                    "h1":float(d.get("usd_1h_change",0) or 0),
                    "h24":float(d.get("usd_24h_change",0) or 0)}
    except: pass
    return None

@st.cache_data(ttl=50)
def get_ohlc(cid,days=2):
    try:
        r=requests.get(f"{CG}/coins/{cid}/ohlc",
                       params={"vs_currency":"usd","days":days},timeout=10)
        if r.status_code==200 and r.json():
            df=pd.DataFrame(r.json(),columns=["ts","Open","High","Low","Close"])
            df["ts"]=pd.to_datetime(df["ts"],unit="ms",utc=True)
            df.set_index("ts",inplace=True)
            return df.astype(float)
    except: pass
    return None

@st.cache_data(ttl=90)
def get_markets_all():
    ids=",".join([c for c,s in APEX_SCAN_PARES])
    try:
        r=requests.get(f"{CG}/coins/markets",params={
            "vs_currency":"usd","ids":ids,"order":"market_cap_desc",
            "per_page":50,"price_change_percentage":"1h,24h","sparkline":"false",
        },timeout=12)
        return {d["id"]:d for d in r.json()} if r.status_code==200 else {}
    except: return {}


# ============================================================
# CALCULO DE FUERZA + NIVELES AUTOMATICOS
# ============================================================
def calcular_todo(df_ohlc, precio, pos=None):
    """
    Calcula indicadores, fuerza alcista y senales de salida.
    Si pos=None, solo calcula indicadores (para APEX scanner).
    """
    result = {"precio":precio,"fuerza":50,"urgencia":0,
              "estado":"OK","ecol":"#4488ff","senales":[],
              "rsi":50,"adx":0,"bb_p":0.5,"p9":0,"p21":0,
              "vel":0,"dip":0,"dim":0,"trail_sl":0,"max_precio":precio,
              "cruce_macd":False,"cruce_e21":False,"vol_ratio":1,
              "atr_pct":2,"atr_abs":precio*0.02,"score_apex":0}

    if df_ohlc is None or len(df_ohlc)<10:
        return result

    c=df_ohlc["Close"].copy(); c.iloc[-1]=precio
    h=df_ohlc["High"]; l=df_ohlc["Low"]; n=len(c)

    # EMAs
    e9 =c.ewm(span=9, adjust=False).mean()
    e21=c.ewm(span=21,adjust=False).mean()
    e50=c.ewm(span=50,adjust=False).mean()
    e200=c.ewm(span=200,adjust=False).mean()
    e9v=float(e9.iloc[-1]); e21v=float(e21.iloc[-1])
    e50v=float(e50.iloc[-1]); e200v=float(e200.iloc[-1])

    # RSI
    d=c.diff(); g=d.clip(lower=0).ewm(com=13,adjust=False).mean()
    ls=(-d.clip(upper=0)).ewm(com=13,adjust=False).mean()
    rsi=float((100-(100/(1+g/(ls+1e-10)))).iloc[-1])
    rsi=50 if np.isnan(rsi) else rsi
    rsi_p=float((100-(100/(1+g/(ls+1e-10)))).iloc[-2]) if n>2 else rsi

    # MACD
    ef=c.ewm(span=12,adjust=False).mean(); es2=c.ewm(span=26,adjust=False).mean()
    mh_s=(ef-es2)-(ef-es2).ewm(span=9,adjust=False).mean()
    mh=float(mh_s.iloc[-1]) if not np.isnan(mh_s.iloc[-1]) else 0
    mhp=float(mh_s.iloc[-2]) if n>2 and not np.isnan(mh_s.iloc[-2]) else mh
    cruce_macd=(mhp<=0 and mh>0)

    # Bollinger
    bm=c.rolling(min(20,n)).mean(); bs=c.rolling(min(20,n)).std()
    bb_p=float(((c-bm+2*bs)/(4*bs+1e-9)).iloc[-1])
    bb_p=0.5 if np.isnan(bb_p) else float(np.clip(bb_p,0,1))

    # ATR y ADX
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr=float(tr.ewm(span=14,adjust=False).mean().iloc[-1])
    atr_pct=atr/precio*100 if precio>0 else 2
    dmp=h.diff().clip(lower=0); dmn=(-l.diff()).clip(lower=0)
    dmp=dmp.where(dmp>dmn,0); dmn=dmn.where(dmn>dmp,0)
    a14=tr.ewm(span=14,adjust=False).mean()
    dip=float((dmp.ewm(span=14,adjust=False).mean()/(a14+1e-10)*100).iloc[-1])
    dim=float((dmn.ewm(span=14,adjust=False).mean()/(a14+1e-10)*100).iloc[-1])
    adx=float(((abs(dip-dim)/(dip+dim+1e-10))*100))

    # Vol ratio
    v=df_ohlc.get("Volume",pd.Series([1]*n))
    vol_r=float(v.iloc[-1]/(v.rolling(20).mean().iloc[-1]+1e-10))

    # Kalman velocidad
    pk=c.ewm(span=5,adjust=False).mean()
    vel=float(pk.diff().iloc[-1])
    velp=float(pk.diff().iloc[-2]) if n>2 else vel

    # Pendientes EMA
    p9 =(e9.iloc[-1] -e9.iloc[-5]) /(e9.iloc[-5]+1e-10)*100  if n>5 else 0
    p21=(e21.iloc[-1]-e21.iloc[-5])/(e21.iloc[-5]+1e-10)*100 if n>5 else 0
    cruce_e21=(c.iloc[-2]<=e21.iloc[-2] and precio>e21v)

    # ---- SCORE APEX ----
    sc=0
    # F1 Tendencia (0-30)
    if precio>e200v: sc+=10
    if e9v>e21v>e50v: sc+=15
    elif e9v>e21v: sc+=8
    if adx>25: sc+=5
    # F2 Impulso (0-35)
    if cruce_macd: sc+=15
    elif mh>0 and mh>mhp: sc+=10
    elif mh>0: sc+=5
    if vel>0 and vel>velp: sc+=10
    elif vel>0: sc+=5
    if vol_r>2: sc+=10
    elif vol_r>1.5: sc+=7
    elif vol_r>1.2: sc+=4
    # F3 Entrada (0-35)
    if 28<=rsi<=55 and rsi>rsi_p: sc+=15
    elif 28<=rsi<=60: sc+=8
    elif rsi>70: sc-=15
    if bb_p<0.2: sc+=12
    elif bb_p<0.4: sc+=7
    elif bb_p>0.85: sc-=12
    if cruce_e21: sc+=8
    elif e9v>e21v and precio>e9v: sc+=4

    sc=max(0,min(100,sc))

    # ---- FUERZA alcista ----
    fuerza=0
    if p9>0:  fuerza+=min(22,p9*55)
    if p21>0: fuerza+=min(18,p21*80)
    if adx>25: fuerza+=22
    elif adx>15: fuerza+=12
    if dip>dim: fuerza+=min(13,(dip-dim)*0.5)
    if vel>0 and vel>velp: fuerza+=10
    elif vel>0: fuerza+=5
    if rsi<65: fuerza+=5
    elif rsi>74: fuerza-=15
    if bb_p>0.85: fuerza-=12
    if mh<0 and mhp>=0: fuerza-=20
    fuerza=max(0,min(100,fuerza))

    # ---- Senales de salida (solo si hay posicion) ----
    senales=[]; urgencia=0
    if pos:
        tsl=float(pos.get("trail_sl",pos.get("sl",precio*0.97)))
        max_p=max(float(pos.get("max_precio",precio)),precio)
        # actualizar trail
        nuevo_sl=precio-float(pos.get("trail",atr*1.5))
        tsl=max(tsl,nuevo_sl)
        if precio<=tsl:
            senales.append("STOP LOSS TOCADO"); urgencia=3
        if rsi>75 and bb_p>0.87:
            senales.append("RSI sobrecomprado + BB techo"); urgencia=max(urgencia,3)
        if e9v<e21v:
            senales.append("EMA9 cruzo EMA21 abajo"); urgencia=max(urgencia,3)
        if mh<0 and mhp>=0:
            senales.append("MACD cruce bajista"); urgencia=max(urgencia,3)
        if rsi>68 and bb_p>0.72:
            senales.append("RSI+BB elevados"); urgencia=max(urgencia,2)
        if fuerza<25:
            senales.append("Fuerza agotada"); urgencia=max(urgencia,2)
        result["trail_sl"]=tsl; result["max_precio"]=max_p

    if urgencia>=3 or fuerza<20:
        estado="SALIR"; ecol="#ff3355"
    elif urgencia==2 or fuerza<40:
        estado="ALERTA"; ecol="#ffaa00"
    else:
        estado="FUERTE"; ecol="#00ff88"

    result.update({
        "fuerza":round(fuerza,1),"urgencia":urgencia,
        "estado":estado,"ecol":ecol,"senales":senales,
        "rsi":round(rsi,1),"adx":round(adx,1),"bb_p":round(bb_p,3),
        "p9":round(p9,3),"p21":round(p21,3),"vel":round(vel,6),
        "dip":round(dip,1),"dim":round(dim,1),
        "vol_ratio":round(vol_r,2),"atr_pct":round(atr_pct,2),
        "atr_abs":atr,"score_apex":sc,
        "cruce_macd":cruce_macd,"cruce_e21":cruce_e21,
        "e9":e9v,"e21":e21v,"e200":e200v,
    })
    return result


def niveles_automaticos(precio, atr, rr=2.5):
    """Calcula SL y TPs automaticamente desde precio y ATR."""
    sl  = precio - 1.4*atr
    tp1 = precio + rr*1.4*atr
    tp2 = precio + rr*2.5*atr
    tp3 = precio + rr*4.0*atr
    trail=1.4*atr
    return sl,tp1,tp2,tp3,trail


# ============================================================
# SIDEBAR - Buscador + Agregar posicion
# ============================================================
st.sidebar.markdown("## AI.LINO")
st.sidebar.markdown("**APEX · Seguimiento · Alta Precision**")
st.sidebar.divider()

# Buscador de cripto
st.sidebar.markdown("**Buscar cripto:**")
q=st.sidebar.text_input("","",placeholder="BTC, SOL, ETH...",
                         key="buscar_cripto",label_visibility="collapsed")

cid_sel=None; sym_sel=None

if q and len(q)>=2:
    q_up=q.upper().strip()
    # Buscar en lista local
    matches=[(cid,sym) for cid,sym in PARES if q_up in sym or q_up in cid.upper()]
    if matches:
        opciones={f"{sym}": cid for cid,sym in matches[:8]}
        elegido=st.sidebar.selectbox("",list(opciones.keys()),
                                      label_visibility="collapsed",
                                      key="sel_cripto")
        cid_sel=opciones[elegido]; sym_sel=elegido
        pr_preview=get_precio(cid_sel)
        if pr_preview:
            p_prev=pr_preview["p"]
            pfmt=(f"${p_prev:,.6f}" if p_prev<1 else
                  f"${p_prev:,.4f}" if p_prev<10 else
                  f"${p_prev:,.2f}")
            c1p,c2p=st.sidebar.columns(2)
            c1p.markdown(f"**{sym_sel}**")
            c2p.markdown(f"**{pfmt}**")
            h1c="#00ff88" if pr_preview['h24']>=0 else "#ff3355"
            st.sidebar.markdown(f'<span style="color:{h1c};font-size:0.8rem">{pr_preview["h24"]:+.2f}% 24H</span>',
                               unsafe_allow_html=True)

st.sidebar.divider()

# Formulario de entrada
st.sidebar.markdown("**Agregar posicion:**")

monto_inv=st.sidebar.number_input(
    "Monto a invertir (USD):",
    min_value=0.0, value=0.0, step=10.0,
    format="%.2f", key="monto_inv"
)

# Si hay precio disponible, calcular cantidad automatica
precio_entry_auto=0.0
if cid_sel:
    pr_now=get_precio(cid_sel)
    if pr_now: precio_entry_auto=pr_now["p"]

precio_entrada=st.sidebar.number_input(
    "Precio entrada:",
    min_value=0.0,
    value=float(precio_entry_auto) if precio_entry_auto>0 else 0.0,
    format="%.6f", key="precio_entrada_inp"
)

# Mostrar cantidad de monedas
if precio_entrada>0 and monto_inv>0:
    cant=monto_inv/precio_entrada
    st.sidebar.caption(f"Cantidad: {cant:.6f} {sym_sel or ''}")

# RR selector
rr_sel=st.sidebar.select_slider(
    "Objetivo R/R:",
    options=[1.5, 2.0, 2.5, 3.0, 4.0],
    value=2.5, key="rr_sel"
)

# Preview de niveles automaticos
if precio_entrada>0 and cid_sel:
    df_prev=get_ohlc(cid_sel,2)
    if df_prev is not None:
        r_prev=calcular_todo(df_prev,precio_entrada)
        atr_prev=r_prev["atr_abs"]
        sl_p,tp1_p,tp2_p,tp3_p,trail_p=niveles_automaticos(
            precio_entrada,atr_prev,rr_sel)
        st.sidebar.markdown("**Niveles calculados:**")
        st.sidebar.markdown(f"""
        <div style="font-family:monospace;font-size:0.78rem;
                    background:#0d1a2e;border-radius:6px;padding:8px">
        <span style="color:#ff3355">SL: {fp(sl_p)}</span><br>
        <span style="color:#ffaa00">TP1: {fp(tp1_p)}</span><br>
        <span style="color:#ffdd44">TP2: {fp(tp2_p)}</span><br>
        <span style="color:#ffffff">TP3: {fp(tp3_p)}</span><br>
        <span style="color:#4488ff">ATR: {r_prev['atr_pct']:.2f}%</span>
        </div>""", unsafe_allow_html=True)

# Boton agregar
if st.sidebar.button("AGREGAR AL SEGUIMIENTO",
                     use_container_width=True,
                     type="primary",
                     key="btn_agregar"):
    if not sym_sel:
        st.sidebar.error("Elige una cripto")
    elif precio_entrada<=0:
        st.sidebar.error("Ingresa precio de entrada")
    elif monto_inv<=0:
        st.sidebar.error("Ingresa monto a invertir")
    else:
        df_ag=get_ohlc(cid_sel,2)
        atr_ag=precio_entrada*0.02
        if df_ag is not None:
            r_ag=calcular_todo(df_ag,precio_entrada)
            atr_ag=r_ag["atr_abs"]
        sl_ag,tp1_ag,tp2_ag,tp3_ag,trail_ag=niveles_automaticos(
            precio_entrada,atr_ag,rr_sel)
        nueva_pos={
            "sym":sym_sel,"cid":cid_sel,
            "entry":precio_entrada,"sl":sl_ag,
            "tp1":tp1_ag,"tp2":tp2_ag,"tp3":tp3_ag,
            "trail":trail_ag,"trail_sl":sl_ag,
            "max_precio":precio_entrada,
            "monto":monto_inv,
            "atr_pct":round(atr_ag/precio_entrada*100,2),
            "score":0,"activa":True,
        }
        db_guardar(nueva_pos)
        st.session_state["_pos_cargada"]=nueva_pos
        st.sidebar.success(f"{sym_sel} agregado")
        time.sleep(0.3); st.rerun()

st.sidebar.divider()

# Auto-update checkbox
auto_up=st.sidebar.checkbox("Auto-update 60 seg",
                             value=st.session_state.get("_auto_up",False),
                             key="_auto_up")

# Boton ejecutar APEX
st.sidebar.divider()
st.sidebar.markdown("**APEX Scanner:**")
tf_apex_ui=st.sidebar.selectbox(
    "Timeframe:",["1D · 30 dias","4H · 10 dias","1D · 90 dias"],
    key="tf_apex_ui"
)
btn_apex=st.sidebar.button("BUSCAR MEJORES ENTRADAS",
                           use_container_width=True,
                           type="primary", key="btn_apex_main")
st.sidebar.caption("Solo educativo. No asesoria financiera.")


# ============================================================
# MAIN - Cargar posicion
# ============================================================
if "_pos_cargada" not in st.session_state:
    st.session_state["_pos_cargada"]=db_cargar()

pos=st.session_state.get("_pos_cargada")


# ============================================================
# SECCION 1: SEGUIMIENTO EN TIEMPO REAL
# ============================================================
st.markdown("## Seguimiento en Tiempo Real")

if not pos or not pos.get("activa",True):
    st.info("Sin posicion activa. Busca una cripto en el sidebar, "
            "ingresa tu monto y precio de entrada, luego haz clic en "
            "'AGREGAR AL SEGUIMIENTO'.")
else:
    sym_p = pos["sym"]; cid_p = pos["cid"]
    entry = float(pos["entry"]); monto  = float(pos.get("monto",0))
    tp1   = float(pos["tp1"]);   tp2    = float(pos["tp2"])
    tp3   = float(pos["tp3"]);   tsl    = float(pos.get("trail_sl",pos["sl"]))

    # Obtener precio actual
    pd_rt=get_precio(cid_p)
    precio=(pd_rt["p"] if pd_rt else entry)
    chg1h =(pd_rt["h1"]  if pd_rt else 0)
    chg24h=(pd_rt["h24"] if pd_rt else 0)

    # Calcular indicadores
    df_rt=get_ohlc(cid_p,2)
    est=calcular_todo(df_rt,precio,pos)

    # Actualizar trail SL si sube
    if est.get("trail_sl",tsl)>tsl:
        tsl=est["trail_sl"]
        db_update_trail(sym_p,tsl,est.get("max_precio",precio))
        pos["trail_sl"]=tsl; pos["max_precio"]=est.get("max_precio",precio)
        st.session_state["_pos_cargada"]=pos

    # Calcular PnL
    pnl_pct=(precio-entry)/(entry+1e-10)*100
    pnl_usd=monto*pnl_pct/100 if monto>0 else 0
    pnl_col="#00ff88" if pnl_pct>=0 else "#ff3355"
    pfmt=(f"${precio:,.6f}" if precio<1 else
          f"${precio:,.4f}" if precio<10 else
          f"${precio:,.2f}")

    fuerza=est["fuerza"]; ecol=est["ecol"]
    urgencia=est["urgencia"]
    tp1_ok=precio>=tp1; tp2_ok=precio>=tp2; tp3_ok=precio>=tp3
    sl_ok =precio<=tsl

    # Alerta de salida
    if sl_ok:
        st.error("STOP LOSS TOCADO - SALIR INMEDIATAMENTE")
    elif tp3_ok:
        st.success("TP3 ALCANZADO - Cierra posicion completa")
    elif tp2_ok:
        st.success("TP2 ALCANZADO - Cierra 75%, deja el resto al TP3")
    elif tp1_ok and urgencia<2:
        st.info("TP1 ALCANZADO - Mueve SL a precio de entrada")
    elif urgencia==3:
        st.error(f"SALIR AHORA - {' | '.join(est['senales'])}")
    elif urgencia==2:
        st.warning(f"PREPARAR SALIDA - {' | '.join(est['senales'])}")

    # Header de posicion
    h_col1,h_col2,h_col3=st.columns([2,1,1])
    with h_col1:
        st.markdown(f"""
        <div style="background:#001a0a;border:2px solid {ecol};
                    border-radius:10px;padding:10px 16px">
            <span style="font-family:'Orbitron',monospace;color:{ecol};
                         font-size:1.2rem;font-weight:700">{sym_p}/USDT</span>
            <span style="color:#c0d8ff;font-size:1.4rem;font-weight:700;
                         margin-left:16px">{pfmt}</span>
            <span style="color:{'#00ff88' if chg24h>=0 else '#ff3355'};
                         font-size:0.85rem;margin-left:8px">{chg24h:+.2f}% 24H</span>
        </div>""", unsafe_allow_html=True)
    with h_col2:
        if st.button("Actualizar datos", key="upd_now"):
            get_precio.clear(); get_ohlc.clear(); st.rerun()
    with h_col3:
        if st.button("Cerrar posicion", key="cerrar_btn"):
            db_cerrar(sym_p,pnl_pct)
            st.session_state["_pos_cargada"]=None
            st.success(f"Cerrado. PnL: {pnl_pct:+.2f}%")
            time.sleep(1); st.rerun()

    # Metricas principales
    m1,m2,m3,m4=st.columns(4)
    m1.metric("PnL %", f"{pnl_pct:+.2f}%",
              f"${pnl_usd:+.2f}" if monto>0 else "")
    m2.metric("Fuerza", f"{fuerza:.0f}%", est["estado"])
    m3.metric("Trail SL", fp(tsl),
              f"Entry: {fp(entry)}")
    m4.metric("Max precio", fp(est.get("max_precio",entry)),
              f"Monto: ${monto:.2f}")

    # Barra de fuerza
    fc="#00ff88" if fuerza>=60 else("#ffaa00" if fuerza>=35 else "#ff3355")
    st.markdown(f"""
    <div style="background:#08090f;border:1.5px solid {ecol};
                border-radius:10px;padding:12px 16px;margin:6px 0">
        <div style="display:flex;justify-content:space-between;margin-bottom:6px">
            <span style="color:{ecol};font-weight:700;font-size:0.9rem">
                FUERZA ALCISTA</span>
            <span style="color:{ecol};font-size:1.4rem;font-weight:700">
                {fuerza:.0f}% {est['estado']}</span>
        </div>
        <div style="background:#0d1428;border-radius:6px;height:20px;overflow:hidden">
            <div style="width:{fuerza}%;height:100%;background:{fc}"></div>
        </div>
        <div style="display:grid;grid-template-columns:repeat(7,1fr);
                    gap:2px;margin-top:8px;font-size:0.72rem;
                    font-family:'Share Tech Mono',monospace">
            <div style="text-align:center;color:#4a6060">RSI<br>
                <b style="color:#ff8844">{est['rsi']}</b></div>
            <div style="text-align:center;color:#4a6060">ADX<br>
                <b style="color:#44aaff">{est['adx']}</b></div>
            <div style="text-align:center;color:#4a6060">BB%<br>
                <b style="color:#ffaa00">{est['bb_p']}</b></div>
            <div style="text-align:center;color:#4a6060">EMA9<br>
                <b style="color:{'#00ff88' if est['p9']>0 else '#ff3355'}">
                {'UP' if est['p9']>0 else 'DN'}</b></div>
            <div style="text-align:center;color:#4a6060">EMA21<br>
                <b style="color:{'#00ff88' if est['p21']>0 else '#ff3355'}">
                {'UP' if est['p21']>0 else 'DN'}</b></div>
            <div style="text-align:center;color:#4a6060">DI<br>
                <b style="color:{'#00ff88' if est['dip']>est['dim'] else '#ff3355'}">
                {'DI+' if est['dip']>est['dim'] else 'DI-'}</b></div>
            <div style="text-align:center;color:#4a6060">Vel<br>
                <b style="color:{'#00ff88' if est['vel']>0 else '#ff3355'}">
                {'UP' if est['vel']>0 else 'DN'}</b></div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Niveles de gestion
    n1,n2,n3,n4,n5=st.columns(5)
    def niv(col,lbl,precio_niv,actual,col_txt,ok=False):
        dist=(precio_niv-actual)/(actual+1e-10)*100
        col.markdown(f"""
        <div style="background:#08090f;border:1px solid {'#00ff88' if ok else '#1a2a4a'};
                    border-radius:8px;padding:8px;text-align:center">
            <div style="color:#4a6060;font-size:0.65rem">{lbl}</div>
            <div style="color:{col_txt};font-weight:700;font-size:0.85rem">
                {fp(precio_niv)}</div>
            <div style="color:{'#00ff88' if ok else '#4a6060'};font-size:0.7rem">
                {'ALCANZADO' if ok else f'{dist:+.1f}%'}</div>
        </div>""", unsafe_allow_html=True)
    niv(n1,"Stop Loss",tsl,precio,"#ff3355",sl_ok)
    niv(n2,"Entry",entry,precio,"#ffffff")
    niv(n3,"TP1",tp1,precio,"#ffaa00",tp1_ok)
    niv(n4,"TP2",tp2,precio,"#ffdd44",tp2_ok)
    niv(n5,"TP3",tp3,precio,"#ffffff",tp3_ok)

    # Senales activas
    if est["senales"]:
        for sig in est["senales"]:
            c="#ff3355" if any(x in sig for x in ["STOP","EMA9","MACD","sobrecomprado"]) else "#ffaa00"
            st.markdown(f'<div style="color:{c};font-family:monospace;'
                       f'font-size:0.8rem;padding:3px 0">{sig}</div>',
                       unsafe_allow_html=True)

    st.divider()

    # Auto-update
    if auto_up:
        time.sleep(60)
        get_precio.clear(); get_ohlc.clear()
        st.rerun()


# ============================================================
# SECCION 2: APEX SCANNER
# ============================================================
st.markdown("## APEX - Mejores Entradas")
st.caption("Solo los pares con mayor potencial: ATR alto + Score alto + R/R optimo")

APEX_TF_MAP={
    "1D · 30 dias": ("1d",30),
    "4H · 10 dias": ("4h",10),
    "1D · 90 dias": ("1d",90),
}

if btn_apex:
    iv_ap,d_ap=APEX_TF_MAP[tf_apex_ui]
    prog=st.progress(0)

    # Obtener precios del mercado
    mkts=get_markets_all()
    resultados=[]
    total=len(APEX_SCAN_PARES)

    for i,(cid,sym) in enumerate(APEX_SCAN_PARES):
        prog.progress((i+1)/total, text=f"Analizando {sym}...")
        try:
            mkt=mkts.get(cid,{})
            precio_mkt=float(mkt.get("current_price",0) or 0)
            if precio_mkt<=0: continue

            chg24=float(mkt.get("price_change_percentage_24h",0) or 0)
            chg1h=float(mkt.get("price_change_percentage_1h_in_currency",0) or 0)

            # OHLC para indicadores
            df_ap=get_ohlc(cid, d_ap if d_ap<=7 else 7)
            time.sleep(0.3)
            if df_ap is None or len(df_ap)<10: continue

            r=calcular_todo(df_ap, precio_mkt)
            sc=r["score_apex"]
            atr_p=r["atr_pct"]

            # Filtro alto impacto
            if sc<55: continue
            if atr_p<1.5: continue     # sin movimiento suficiente
            if r["adx"]<15: continue   # sin tendencia
            if r["rsi"]>72: continue   # ya sobrecomprado

            # Calcular niveles
            sl_a,tp1_a,tp2_a,tp3_a,tr_a=niveles_automaticos(
                precio_mkt,r["atr_abs"],2.5)
            rr=(tp1_a-precio_mkt)/(precio_mkt-sl_a+1e-10)

            pfmt=(f"${precio_mkt:,.6f}" if precio_mkt<1 else
                  f"${precio_mkt:,.4f}" if precio_mkt<10 else
                  f"${precio_mkt:,.2f}")

            resultados.append({
                "cid":cid,"sym":sym,"precio":precio_mkt,"pfmt":pfmt,
                "score":sc,"atr_pct":atr_p,"fuerza":r["fuerza"],
                "rsi":r["rsi"],"adx":r["adx"],"vol":r["vol_ratio"],
                "chg1h":chg1h,"chg24":chg24,
                "sl":sl_a,"tp1":tp1_a,"tp2":tp2_a,"tp3":tp3_a,
                "trail":tr_a,"rr":round(rr,2),
                "cruce_macd":r["cruce_macd"],"cruce_e21":r["cruce_e21"],
                "impacto":round(sc*atr_p*rr/100,2),
            })
        except: pass

    prog.empty()

    # Ordenar por impacto
    resultados.sort(key=lambda x:x["impacto"],reverse=True)

    if not resultados:
        st.info("Sin senales de alto impacto ahora. El mercado no esta en condiciones optimas.")
    else:
        st.caption(f"{len(resultados)} oportunidades encontradas — ordenadas por impacto")

        # Top cards
        top=resultados[:6]
        for row_start in range(0,min(6,len(top)),3):
            row=top[row_start:row_start+3]
            cols=st.columns(len(row))
            for r,col in zip(row,cols):
                sc_col=("#00ff88" if r["score"]>=75 else
                        "#ffaa00" if r["score"]>=60 else "#4488ff")
                with col:
                    st.markdown(f"""
                    <div style="background:#08090f;border:1.5px solid {sc_col};
                                border-radius:10px;padding:12px">
                        <div style="font-family:'Orbitron',monospace;
                                    color:{sc_col};font-weight:700;font-size:1rem">
                            {r['sym']}/USDT</div>
                        <div style="font-size:1.2rem;font-weight:700;
                                    color:#c0d8ff;margin:4px 0">{r['pfmt']}</div>
                        <div style="color:{'#00ff88' if r['chg24']>=0 else '#ff3355'};
                                    font-size:0.8rem">{r['chg24']:+.2f}% 24H</div>
                        <div style="display:grid;grid-template-columns:1fr 1fr;
                                    gap:4px;margin:8px 0;font-size:0.72rem;
                                    font-family:'Share Tech Mono',monospace">
                            <div>Score: <b style="color:{sc_col}">{r['score']}/100</b></div>
                            <div>ATR: <b style="color:#ffaa00">{r['atr_pct']}%</b></div>
                            <div>R/R: <b style="color:#00ff88">{r['rr']}x</b></div>
                            <div>ADX: <b style="color:#44aaff">{r['adx']}</b></div>
                            <div>RSI: <b style="color:#ff8844">{r['rsi']}</b></div>
                            <div>Vol: <b style="color:#4488ff">{r['vol']}x</b></div>
                        </div>
                        <div style="border-top:1px solid #0d1a2e;padding-top:6px;
                                    font-size:0.7rem;font-family:'Share Tech Mono',monospace">
                            <div style="color:#ff3355">SL: {fp(r['sl'])}</div>
                            <div style="color:#ffaa00">TP1: {fp(r['tp1'])}</div>
                            <div style="color:#ffdd44">TP2: {fp(r['tp2'])}</div>
                            <div style="color:#ffffff">TP3: {fp(r['tp3'])}</div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                    # Boton agregar directo
                    import uuid as _u
                    if st.button(f"Agregar {r['sym']}",
                                key=f"ag_{r['sym']}_{str(_u.uuid4())[:6]}"):
                        # Usar el monto del sidebar si fue ingresado
                        monto_ag=st.session_state.get("monto_inv",0)
                        nueva={
                            "sym":r["sym"],"cid":r["cid"],
                            "entry":r["precio"],"sl":r["sl"],
                            "tp1":r["tp1"],"tp2":r["tp2"],"tp3":r["tp3"],
                            "trail":r["trail"],"trail_sl":r["sl"],
                            "max_precio":r["precio"],"monto":monto_ag,
                            "atr_pct":r["atr_pct"],"score":r["score"],
                            "activa":True,
                        }
                        db_guardar(nueva)
                        st.session_state["_pos_cargada"]=nueva
                        st.success(f"{r['sym']} agregado al seguimiento")
                        time.sleep(0.5); st.rerun()

        # Tabla completa
        if len(resultados)>3:
            with st.expander(f"Ver todos ({len(resultados)})"):
                filas=[{
                    "Par":r["sym"]+"/USDT",
                    "Score":r["score"],"Impacto":r["impacto"],
                    "ATR%":r["atr_pct"],"R/R":r["rr"],
                    "RSI":r["rsi"],"ADX":r["adx"],"Vol":r["vol"],
                    "SL":fp(r["sl"]),"TP1":fp(r["tp1"]),"TP3":fp(r["tp3"]),
                    "MACD":"Si" if r["cruce_macd"] else "-",
                    "EMA21":"Si" if r["cruce_e21"] else "-",
                } for r in resultados]
                st.dataframe(pd.DataFrame(filas),
                           use_container_width=True,hide_index=True,
                           column_config={
                               "Score":st.column_config.ProgressColumn(
                                   "Score",min_value=0,max_value=100,format="%d"),
                               "Impacto":st.column_config.NumberColumn(
                                   "Impacto",format="%.1f"),
                           })
