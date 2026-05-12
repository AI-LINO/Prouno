import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")          # backend no-interactivo — evita conflictos en Streamlit Cloud
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from hmmlearn import hmm
import requests
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI.Lino Quantum Engine", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Rajdhani',sans-serif}
.main{background-color:#050810}
.stSidebar{background-color:#08090f;border-right:1px solid #0d1a2e}
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
 font-weight:700;font-size:0.78rem;letter-spacing:2px;padding:0.7rem 1rem;
 width:100%;transition:all 0.3s}
.stButton>button:hover{background:linear-gradient(135deg,#0022aa,#0066ff,#00aaff);
 color:white;transform:translateY(-2px);box-shadow:0 4px 20px #0044ff44}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  TIMEFRAMES
# ══════════════════════════════════════════════════════════════
TIMEFRAMES = {
    "1H  · 3 días"  : ("1h",  "1h",   3,   "3 días · 1H"),
    "4H  · 10 días" : ("4h",  "1h",   10,  "10 días · 4H"),
    "1D  · 1 mes"   : ("1d",  "1d",   30,  "1 mes · 1D"),
    "1D  · 3 meses" : ("1d",  "1d",   90,  "3 meses · 1D"),
    "1D  · 6 meses" : ("1d",  "1d",   180, "6 meses · 1D"),
    "1W  · 1 año"   : ("1d",  "1wk",  365, "1 año · 1W"),
}

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
    q=q.upper(); todos=binance_get_all_symbols()
    prio={"USDT":0,"BTC":1,"ETH":2,"BNB":3}
    found=[p for p in todos if q in p["base"] or q in p["symbol"]]
    found.sort(key=lambda x:prio.get(x["quote"],9))
    return found[:8]

def binance_descargar(symbol, interval_bn, dias):
    start_ms=int((datetime.utcnow()-timedelta(days=dias)).timestamp()*1000)
    url="https://api.binance.com/api/v3/klines"
    params={"symbol":symbol,"interval":interval_bn,"startTime":start_ms,"limit":1000}
    filas=[]
    for _ in range(5):
        r=requests.get(url,params=params,timeout=15); data=r.json()
        if not data or isinstance(data,dict): break
        filas.extend(data)
        if len(data)<1000: break
        params["startTime"]=data[-1][0]+1
    if not filas: return pd.DataFrame()
    df=pd.DataFrame(filas,columns=["Open time","Open","High","Low","Close","Volume",
        "Close time","qav","trades","tbb","tbq","ignore"])
    df["Open time"]=pd.to_datetime(df["Open time"],unit="ms",utc=True)
    df.set_index("Open time",inplace=True)
    for col in ["Open","High","Low","Close","Volume"]: df[col]=df[col].astype(float)
    return df[["Open","High","Low","Close","Volume"]]

CG_BASE="https://api.coingecko.com/api/v3"

def coingecko_buscar(query):
    try:
        r=requests.get(f"{CG_BASE}/search",params={"query":query},timeout=10)
        return r.json().get("coins",[])[:6]
    except: return []

def coingecko_descargar(coin_id, dias):
    url=f"{CG_BASE}/coins/{coin_id}/market_chart"
    params={"vs_currency":"usd","days":dias,"interval":"daily" if dias>3 else "hourly"}
    try:
        r=requests.get(url,params=params,timeout=15)
        if r.status_code==429: time.sleep(30); r=requests.get(url,params=params,timeout=15)
        data=r.json()
        if "prices" not in data: return pd.DataFrame()
        prices=pd.DataFrame(data["prices"],columns=["ts","Close"])
        volumes=pd.DataFrame(data["total_volumes"],columns=["ts","Volume"])
        df=prices.merge(volumes,on="ts")
        df["ts"]=pd.to_datetime(df["ts"],unit="ms",utc=True)
        df.set_index("ts",inplace=True)
        df["Open"]=df["Close"].shift(1).fillna(df["Close"])
        df["High"]=df["Close"]; df["Low"]=df["Close"]
        return df[["Open","High","Low","Close","Volume"]]
    except: return pd.DataFrame()

def yahoo_descargar(ticker, interval_yf, dias):
    pm={3:"5d",10:"1mo",30:"1mo",90:"3mo",180:"6mo",365:"1y"}
    t=yf.Ticker(ticker); df=t.history(period=pm.get(dias,"1y"),interval=interval_yf)
    return df if not df.empty else pd.DataFrame()

def cargar_datos(ticker, fuente, tf_key):
    ibn,iyf,dias,_=TIMEFRAMES[tf_key]
    if fuente=="binance":    return binance_descargar(ticker,ibn,dias)
    elif fuente=="coingecko": return coingecko_descargar(ticker,dias)
    else:                    return yahoo_descargar(ticker,iyf,dias)

def fp(v):
    if v is None or (isinstance(v,float) and np.isnan(v)): return "—"
    return f"${v:,.6f}" if abs(v)<1 else f"${v:,.4f}" if abs(v)<10 else f"${v:,.2f}"

# ══════════════════════════════════════════════════════════════
#  INDICADORES TÉCNICOS
# ══════════════════════════════════════════════════════════════
def calcular_rsi(c, p=14):
    d=c.diff(); g=d.clip(lower=0).ewm(com=p-1,adjust=False).mean()
    l=(-d.clip(upper=0)).ewm(com=p-1,adjust=False).mean()
    return 100-(100/(1+g/(l+1e-10)))

def calcular_macd(c, f=12, s=26, sig=9):
    ef=c.ewm(span=f,adjust=False).mean(); es=c.ewm(span=s,adjust=False).mean()
    m=ef-es; sl=m.ewm(span=sig,adjust=False).mean(); return m,sl,m-sl

def calcular_bb(c, p=20, k=2):
    sma=c.rolling(p).mean(); std=c.rolling(p).std()
    up=sma+k*std; lo=sma-k*std
    return up,sma,lo,(c-lo)/(up-lo+1e-9)

def calcular_atr(h, l, c, p=14):
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(span=p,adjust=False).mean()   # EMA-ATR más reactivo

def calcular_stoch_rsi(c, p=14, sk=3, sd=3):
    rsi=calcular_rsi(c,p); mn=rsi.rolling(p).min(); mx=rsi.rolling(p).max()
    st=(rsi-mn)/(mx-mn+1e-9)*100
    return st.rolling(sk).mean(), st.rolling(sk).mean().rolling(sd).mean()

def calcular_indicadores(df):
    c=df["Close"]; ind={}
    ind["rsi"]=calcular_rsi(c)
    ind["macd"],ind["macd_sig"],ind["macd_hist"]=calcular_macd(c)
    ind["bb_up"],ind["bb_mid"],ind["bb_lo"],ind["bb_pct"]=calcular_bb(c)
    ind["atr"]=calcular_atr(df["High"],df["Low"],c)
    ind["atr_pct"]=ind["atr"]/c*100
    ind["ema9"]=c.ewm(span=9,adjust=False).mean()
    ind["ema21"]=c.ewm(span=21,adjust=False).mean()
    ind["ema50"]=c.ewm(span=50,adjust=False).mean()
    ind["stoch_k"],ind["stoch_d"]=calcular_stoch_rsi(c)
    tp=(df["High"]+df["Low"]+c)/3
    ind["vwap"]=(tp*df["Volume"]).cumsum()/df["Volume"].cumsum()
    ind["vol_sma"]=df["Volume"].rolling(20).mean()
    ind["vol_ratio"]=df["Volume"]/(ind["vol_sma"]+1e-10)
    ind["momentum"]=c.pct_change(5)*100
    ind["ret_log"]=np.log(c/(c.shift(1)+1e-10))
    ind["vol_ann"]=ind["ret_log"].rolling(15).std()*np.sqrt(252)*100
    # Cambio de precio últimas N velas (señal rápida)
    ind["chg1"]=c.pct_change(1)*100
    ind["chg3"]=c.pct_change(3)*100
    return ind

# ══════════════════════════════════════════════════════════════
#  HMM MEJORADO — Robusto para cripto volátil y acciones
# ══════════════════════════════════════════════════════════════
def entrenar_hmm(df, ind):
    """
    Features de 5 dimensiones para mejor separación de regímenes:
    1. Retorno log normalizado
    2. Volatilidad realizada (rolling std)
    3. Momentum 5 períodos
    4. RSI normalizado (0-1)
    5. Posición en Bollinger (%B)
    
    Mejoras vs versión anterior:
    - EMA-ATR en lugar de SMA-ATR (más reactivo en cripto)
    - RSI con EWM en lugar de SMA (responde más rápido)
    - Regularización de covarianza (evita colapso en activos muy volátiles)
    - n_iter=3000, múltiples inicializaciones → mejor convergencia
    - Selección por BIC + AIC combinados
    """
    ret  = ind["ret_log"].dropna()
    vol  = ind["vol_ann"].dropna()
    mom  = ind["momentum"].dropna()
    rsi  = (ind["rsi"].dropna() / 100.0)
    pctb = ind["bb_pct"].clip(0,1).dropna()

    feat = pd.concat([ret, vol, mom, rsi, pctb], axis=1).dropna()
    feat.columns = ["r","v","m","rsi","pctb"]

    if len(feat) < 25:
        # Fallback a 2 features si hay muy pocos datos (cripto nueva)
        feat = feat[["r","v"]]

    # Normalización robusta (mediana + IQR) — más estable en cripto
    med  = feat.median()
    iqr  = feat.quantile(0.75) - feat.quantile(0.25) + 1e-8
    X    = ((feat - med) / iqr).values
    X   += np.random.normal(0, 1e-5, X.shape)

    best_m, best_score = None, np.inf
    n_max = min(4, max(2, len(X)//20))

    for n in range(2, n_max+1):
        for seed in [42, 7, 99]:   # múltiples semillas → evitar mínimos locales
            try:
                m = hmm.GaussianHMM(
                    n_components=n,
                    covariance_type="full",
                    n_iter=3000,
                    tol=1e-5,
                    random_state=seed,
                    init_params="stmc",
                    params="stmc"
                )
                m.fit(X)
                # Combinar BIC + penalización por estados muy pequeños
                states_tmp = m.predict(X)
                min_state_frac = np.min(np.bincount(states_tmp)) / len(X)
                penalty = 0 if min_state_frac > 0.08 else 1e6   # penalizar si un estado < 8%
                score = m.bic(X) + penalty
                if score < best_score:
                    best_score, best_m = score, m
            except: pass

    if best_m is None:
        raise ValueError("HMM no convergió")

    states = best_m.predict(X)

    # Etiquetar estados por retorno medio + volatilidad
    means  = best_m.means_[:, 0]
    vols_m = best_m.means_[:, 1] if best_m.means_.shape[1] > 1 else np.zeros(best_m.n_components)
    idx_r  = np.argsort(means)
    lmap   = {}

    lmap[idx_r[0]]  = {"nombre":"PÁNICO / BAJISTA",     "color":"#ff3355","emoji":"🔴","urgencia":4}
    lmap[idx_r[-1]] = {"nombre":"ALCISTA / EUFORIA",    "color":"#00ff88","emoji":"🟢","urgencia":2}
    for i in range(best_m.n_components):
        if i not in lmap:
            if means[i] > 0.002:
                lmap[i] = {"nombre":"ACUMULACIÓN",          "color":"#4488ff","emoji":"🔵","urgencia":1}
            else:
                lmap[i] = {"nombre":"LATERAL/DISTRIBUCIÓN", "color":"#ffaa00","emoji":"🟡","urgencia":3}

    return best_m, states, feat.index, lmap

# ══════════════════════════════════════════════════════════════
#  SCORE COMPUESTO + SEÑAL DE SALIDA RÁPIDA (para cripto)
# ══════════════════════════════════════════════════════════════
def calcular_score(ind, hmm_urgencia=2):
    p=[]
    rsi=ind["rsi"].iloc[-1] if not pd.isna(ind["rsi"].iloc[-1]) else 50

    # RSI (20 pts)
    if rsi<25:    p.append(("RSI Sobreventa extrema",20,20,"🟢"))
    elif rsi<35:  p.append(("RSI Sobreventa",16,20,"🟢"))
    elif rsi<48:  p.append(("RSI Zona baja",12,20,"🟡"))
    elif rsi<55:  p.append(("RSI Neutral",8,20,"⚪"))
    elif rsi<68:  p.append(("RSI Zona alta",5,20,"🟡"))
    elif rsi<78:  p.append(("RSI Sobrecompra",2,20,"🔴"))
    else:         p.append(("RSI Extremo SC",0,20,"🔴"))

    # MACD (18 pts)
    hist=ind["macd_hist"].iloc[-1]; prev=ind["macd_hist"].iloc[-2] if len(ind["macd_hist"])>1 else hist
    if pd.isna(hist): hist=0; prev=0
    if prev<=0 and hist>0:         p.append(("MACD Cruce alcista",18,18,"🟢"))
    elif hist>0 and hist>prev*1.2: p.append(("MACD Acelerando",14,18,"🟢"))
    elif hist>0:                   p.append(("MACD Positivo",9,18,"🟡"))
    elif prev>=0 and hist<0:       p.append(("MACD Cruce bajista",0,18,"🔴"))
    elif hist<prev*1.2:            p.append(("MACD Cayendo",3,18,"🔴"))
    else:                          p.append(("MACD Negativo",5,18,"🔴"))

    # Bollinger %B (12 pts)
    pct=ind["bb_pct"].iloc[-1] if not pd.isna(ind["bb_pct"].iloc[-1]) else 0.5
    if pct<0.0:    p.append(("BB Bajo banda",12,12,"🟢"))
    elif pct<0.25: p.append(("BB Zona baja",9,12,"🟡"))
    elif pct<0.55: p.append(("BB Centro",6,12,"⚪"))
    elif pct<0.85: p.append(("BB Zona alta",3,12,"🟡"))
    else:          p.append(("BB Sobre banda",0,12,"🔴"))

    # EMAs alineación (18 pts)
    e9=ind["ema9"].iloc[-1]; e21=ind["ema21"].iloc[-1]; e50=ind["ema50"].iloc[-1]
    close=ind["bb_mid"].iloc[-1]
    es=0
    if not any(pd.isna(v) for v in [e9,e21,e50,close]):
        if e9>e21>e50 and close>e9:    es=18   # alcista perfecto
        elif e9>e21>e50:               es=14
        elif e9>e21:                   es=10
        elif e9>e50:                   es=6
        else:                          es=2
    p.append(("EMAs alineadas",es,18,"🟢" if es>=14 else("🟡" if es>=8 else "🔴")))

    # Volumen (12 pts)
    vr=ind["vol_ratio"].iloc[-1] if not pd.isna(ind["vol_ratio"].iloc[-1]) else 1
    chg1=ind["chg1"].iloc[-1] if not pd.isna(ind["chg1"].iloc[-1]) else 0
    # Volumen alto en caída = señal de capitulación (compra)
    if vr>2 and chg1<-1:   p.append(("Vol alto+caída=capitulación",10,12,"🟢"))
    elif vr>2 and chg1>0:  p.append(("Vol alto+subida",12,12,"🟢"))
    elif vr>1.5:           p.append(("Vol elevado",8,12,"🟡"))
    elif vr>0.7:           p.append(("Vol normal",6,12,"⚪"))
    else:                  p.append(("Vol seco",2,12,"🔴"))

    # Stoch RSI (10 pts) — peso extra en zona extrema
    sk=ind["stoch_k"].iloc[-1]; sd=ind["stoch_d"].iloc[-1]
    if not any(pd.isna(v) for v in [sk,sd]):
        if sk<15 and sk>sd:    p.append(("StochRSI cruce alcista",10,10,"🟢"))
        elif sk<25:            p.append(("StochRSI S.venta",7,10,"🟡"))
        elif sk>85 and sk<sd:  p.append(("StochRSI cruce bajista",0,10,"🔴"))
        elif sk>75:            p.append(("StochRSI S.compra",2,10,"🟡"))
        else:                  p.append(("StochRSI Neutral",5,10,"⚪"))
    else: p.append(("StochRSI N/D",5,10,"⚪"))

    # Penalización HMM urgencia (estados de pánico/distribución)
    hmm_pen = {1:0, 2:0, 3:-8, 4:-15}.get(hmm_urgencia, 0)

    total_pts = sum(x[1] for x in p) + hmm_pen
    total_max = sum(x[2] for x in p)
    score = max(0, min(100, round(total_pts / total_max * 100)))
    return score, p

def generar_señal(score, ind, hmm_urgencia=2, fuente="yahoo"):
    precio = ind["bb_mid"].iloc[-1]
    atr    = ind["atr"].iloc[-1]
    if pd.isna(atr) or atr==0: atr = precio*0.02

    # Para cripto: SL/TP más ajustados (mayor volatilidad)
    mult = 1.2 if fuente in ("binance","coingecko") else 1.5

    if score >= 72:
        return("🟢 COMPRA / LONG","sc",
               precio-mult*atr, precio+2*mult*atr, precio+3.5*mult*atr,
               "Alta probabilidad alcista. Múltiples indicadores alineados.")
    elif score >= 58:
        return("🟡 VIGILAR — Posible entrada","se",
               precio-1.5*mult*atr, precio+1.5*mult*atr, precio+2.5*mult*atr,
               "Señales mixtas con sesgo alcista. Espera confirmación o reduce tamaño.")
    elif score >= 40:
        return("⚪ NEUTRAL — Sin señal clara","sn",None,None,None,
               "Mercado sin dirección definida. Mejor mantenerse al margen.")
    elif score >= 25:
        return("🔴 PRECAUCIÓN — Sesgo bajista","sv",None,None,None,
               "Señales de debilidad. Si tienes posición, considera SL ajustado.")
    else:
        # Señal de salida urgente — especialmente útil en cripto
        return("🚨 SALIDA URGENTE — Alta presión bajista","sv",
               precio+mult*atr, precio-2*mult*atr, precio-3.5*mult*atr,
               "⚠️ Múltiples indicadores bajistas + régimen de pánico. Protege capital AHORA.")

# ══════════════════════════════════════════════════════════════
#  MÓDULO 1: OSCILADOR ARMÓNICO CUÁNTICO
# ══════════════════════════════════════════════════════════════
def quantum_harmonic_oscillator(prices, n_levels=5):
    c   = np.array(prices, dtype=float)
    N   = len(c)
    mu  = np.mean(c); sigma = np.std(c)+1e-10
    x_n = (c-mu)/sigma
    fft = np.abs(np.fft.rfft(x_n-np.mean(x_n)))
    frq = np.fft.rfftfreq(N)
    dom = frq[np.argmax(fft[1:])+1] if len(fft)>2 else 1/N
    omega = 2*np.pi*max(dom, 1/N)
    E_n  = [(n+0.5)*omega for n in range(n_levels)]
    p_up = [mu+np.sqrt(2*E/(omega**2+1e-10))*sigma for E in E_n]
    p_dn = [mu-np.sqrt(2*E/(omega**2+1e-10))*sigma for E in E_n]
    xg   = np.linspace(x_n.min()-1, x_n.max()+1, 500)
    psi  = np.exp(-xg**2)/np.sqrt(np.pi)
    for n in range(1,4):
        Hn = {1:2*xg, 2:4*xg**2-2, 3:8*xg**3-12*xg}[n]
        psi += np.exp(-E_n[n]*0.5)*Hn**2*np.exp(-xg**2)
    psi /= (psi.sum()+1e-10)
    xc  = x_n[-1]; Vc = 0.5*omega**2*xc**2
    zona = "REBOTE CUÁNTICO" if abs(xc)>1.5 else ("EQUILIBRIO" if abs(xc)<0.5 else "TRANSICIÓN")
    return {"x_grid":xg,"psi_sq":psi,"price_levels_up":p_up,"price_levels_down":p_dn,
            "omega":omega,"mu":mu,"sigma":sigma,"x_current":xc,"V_current":Vc,"zona":zona}

# ══════════════════════════════════════════════════════════════
#  MÓDULO 2: PRINCIPIO DE HEISENBERG
# ══════════════════════════════════════════════════════════════
def heisenberg_uncertainty(prices, ventana=20):
    c = pd.Series(prices)
    ventana = min(ventana, len(c)//3)
    dx = c.rolling(ventana).std()/(c.rolling(ventana).mean()+1e-10)
    ret = np.log(c/(c.shift(1)+1e-10))
    dp  = ret.rolling(ventana).std()*np.sqrt(ventana)
    U   = dx*dp; hbar = U.median()
    Un  = U/(hbar+1e-10)
    clas= pd.Series("INCERTIDUMBRE NORMAL", index=c.index)
    clas[Un<0.7]  = "BAJA INCERTIDUMBRE"
    clas[Un>=1.3] = "ALTA INCERTIDUMBRE"
    zs  = (c-c.rolling(ventana).mean())/(c.rolling(ventana).std()+1e-10)
    tunel = (zs.abs()>2.0)&(Un<1.0)
    ea  = clas.iloc[-1]; ua = Un.iloc[-1]
    confianza = max(0,min(100,int((1.5-min(ua,1.5))/1.5*100)))
    return {"delta_x":dx,"U_norm":Un,"hbar_mkt":hbar,"clasificacion":clas,
            "tunel":tunel,"estado_actual":ea,"confianza":confianza}

# ══════════════════════════════════════════════════════════════
#  MÓDULO 3: FILTRO DE KALMAN
# ══════════════════════════════════════════════════════════════
def kalman_filter(prices):
    z  = np.array(prices, dtype=float); N = len(z)
    A  = np.array([[1,1],[0,1]])
    H  = np.array([[1,0]])
    so = np.std(np.diff(z))*0.5+1e-10
    sp = np.std(np.diff(z))*0.1+1e-10
    R  = np.array([[so**2]])
    Q  = np.array([[sp**2,0],[0,(sp*0.1)**2]])
    xe = np.zeros((N,2)); xe[0]=[z[0],0]
    P  = np.eye(2)*so**2; Pl=[P]
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
            "tendencia":tend,"diff_pct":dp,"vel_actual":va,"kalman_actual":pk[-1]}

# ══════════════════════════════════════════════════════════════
#  MÓDULO 4: ENTRELAZAMIENTO — siempre con datos diarios
# ══════════════════════════════════════════════════════════════
def quantum_entanglement(ticker_main, fuente):
    """Siempre usa datos diarios de 90 días — independiente del timeframe elegido."""
    pares_ref = {
        "BTC-USD":"₿ Bitcoin","ETH-USD":"Ξ Ethereum",
        "SPY":"📊 S&P500","GLD":"🥇 Oro",
        "^VIX":"😰 VIX","DX-Y.NYB":"💵 USD",
    }
    # Obtener retornos del activo principal en diario
    try:
        if fuente=="binance":
            df_m = binance_descargar(ticker_main,"1d",90)
        elif fuente=="coingecko":
            df_m = coingecko_descargar(ticker_main,90)
        else:
            t=yf.Ticker(ticker_main); df_m=t.history(period="3mo",interval="1d")
        if df_m.empty or len(df_m)<15: return None
    except: return None

    ret_main = np.log(df_m["Close"]/(df_m["Close"].shift(1)+1e-10)).dropna()
    # Normalizar índice a date para facilitar intersección
    if hasattr(ret_main.index,'tz') and ret_main.index.tz is not None:
        ret_main.index = ret_main.index.tz_localize(None)
    ret_main.index = pd.to_datetime(ret_main.index).normalize()

    correlaciones={}
    for sym,nombre in pares_ref.items():
        if sym==ticker_main: continue
        try:
            t=yf.Ticker(sym); df2=t.history(period="3mo",interval="1d")
            if df2.empty or len(df2)<15: continue
            ret2=np.log(df2["Close"]/(df2["Close"].shift(1)+1e-10)).dropna()
            if hasattr(ret2.index,'tz') and ret2.index.tz is not None:
                ret2.index=ret2.index.tz_localize(None)
            ret2.index=pd.to_datetime(ret2.index).normalize()
            idx=ret_main.index.intersection(ret2.index)
            if len(idx)<12: continue
            corr=np.corrcoef(ret_main.loc[idx].values,ret2.loc[idx].values)[0,1]
            if not np.isnan(corr):
                correlaciones[nombre]={"sym":sym,"corr":float(corr)}
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
        if co>0.7:    correlaciones[nm]["tipo"]=("ENTRELAZADO ↑↑","#00ff88")
        elif co>0.35: correlaciones[nm]["tipo"]=("CORRELADO ↑","#44aaff")
        elif co>-0.35:correlaciones[nm]["tipo"]=("INDEPENDIENTE ⊥","#aaaaaa")
        elif co>-0.7: correlaciones[nm]["tipo"]=("ANTI-CORR ↑↓","#ffaa00")
        else:         correlaciones[nm]["tipo"]=("ESPEJO CUÁNTICO ↓↑","#ff3355")
    return {"correlaciones":correlaciones,"nombres":nombres,"C_matrix":C,
            "S_vn":Svn,"S_norm":Sn}

# ══════════════════════════════════════════════════════════════
#  FUNCIÓN CENTRAL DE DIBUJO — siempre fuera de st.columns
# ══════════════════════════════════════════════════════════════
BG="#050810"
def mk_fig(*args, **kwargs):
    fig=plt.figure(*args,facecolor=BG,**kwargs)
    return fig

def estilizar_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors="#2a4060",labelsize=7)
    for sp in ax.spines.values(): sp.set_color("#0d1a2e")

def render_fig(fig):
    """Renderiza y cierra — evita memory leak y NULL en Streamlit."""
    st.pyplot(fig); plt.close(fig)

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
st.sidebar.markdown("# 🌌 AI.LINO")
st.sidebar.markdown("**QUANTUM ENGINE v3**")
st.sidebar.divider()

query=st.sidebar.text_input("🔍 Buscar:",placeholder="Monad, BTC, Tesla...",key="sq")
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
    favs={"MON·Monad🦎":("monad","Monad(MON)","coingecko"),
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
tf_key=st.sidebar.selectbox("⏱ Timeframe",list(TIMEFRAMES.keys()),index=2)
st.sidebar.markdown("**⚛️ Módulos:**")
mod_osc=st.sidebar.toggle("1· Oscilador Armónico",value=True)
mod_hei=st.sidebar.toggle("2· Heisenberg",value=True)
mod_kal=st.sidebar.toggle("3· Kalman",value=True)
mod_ent=st.sidebar.toggle("4· Entrelazamiento",value=True)
if ticker_final:
    bdg={"yahoo":"📈Yahoo","binance":"🟡Binance","coingecko":"🦎CoinGecko"}.get(fuente,"")
    st.sidebar.info(f"📌 **{ticker_final}** · {bdg}")
ejecutar=st.sidebar.button("⚛️ ANALIZAR",use_container_width=True)
st.sidebar.caption("⚠️ Solo educativo. No es asesoría financiera.")

# ══════════════════════════════════════════════════════════════
#  EJECUCIÓN
# ══════════════════════════════════════════════════════════════
if ejecutar:
    if not ticker_final: st.error("Selecciona un instrumento."); st.stop()

    _,_,dias,tf_disp=TIMEFRAMES[tf_key]
    bdg_txt={"yahoo":"📈 Yahoo","binance":"🟡 Binance","coingecko":"🦎 CoinGecko"}.get(fuente,"")

    with st.spinner("⬇️ Descargando datos..."):
        df_raw=cargar_datos(ticker_final,fuente,tf_key)

    if df_raw is None or df_raw.empty or len(df_raw)<20:
        st.error("❌ Datos insuficientes. Prueba un timeframe mayor o diferente fuente."); st.stop()

    with st.spinner("🧮 Calculando indicadores..."):
        ind=calcular_indicadores(df_raw)

    with st.spinner("🤖 Entrenando HMM mejorado..."):
        try:
            best_model,states,feat_idx,lmap=entrenar_hmm(df_raw,ind)
            hmm_ok=True
            hmm_urg=lmap[states[-1]]["urgencia"]
        except Exception as e:
            hmm_ok=False; hmm_urg=2
            st.warning(f"HMM no convergió ({e}) — análisis técnico activo.")

    score,desglose=calcular_score(ind, hmm_urg)
    señal,cls,sl,tp1,tp2,desc=generar_señal(score,ind,hmm_urg,fuente)

    precio_actual=df_raw["Close"].iloc[-1]
    cambio_1p=df_raw["Close"].pct_change(1).iloc[-1]*100

    # ── HEADER ──────────────────────────────────────────────
    st.markdown(f"## ⚛️ {ticker_nombre}")
    st.caption(f"{bdg_txt} · {tf_disp} · {len(df_raw)} velas · {datetime.utcnow().strftime('%H:%M UTC')}")

    pfmt=f"${precio_actual:,.6f}" if precio_actual<1 else f"${precio_actual:,.4f}" if precio_actual<10 else f"${precio_actual:,.2f}"
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("Precio",pfmt,f"{cambio_1p:+.2f}%")
    rsi_v=ind["rsi"].iloc[-1]; c2.metric("RSI(14)",f"{rsi_v:.1f}" if not pd.isna(rsi_v) else "N/D")
    av=ind["atr_pct"].iloc[-1]; c3.metric("Volatilidad",f"{av:.2f}%" if not pd.isna(av) else "N/D")
    vr=ind["vol_ratio"].iloc[-1]; c4.metric("Vol Ratio",f"{vr:.2f}x" if not pd.isna(vr) else "N/D")
    sc_col="#00ff88" if score>=65 else("#ffaa00" if score>=45 else "#ff3355")
    c5.metric("Score IA",f"{score}/100")
    st.divider()

    # ── GRÁFICO PRINCIPAL (ANCHO COMPLETO — sin columns alrededor) ──
    plt.style.use("dark_background")
    fig_main=mk_fig(figsize=(14,11))
    gs=gridspec.GridSpec(4,1,figure=fig_main,hspace=0.07,height_ratios=[3,1,1,1])
    ax1=fig_main.add_subplot(gs[0]); ax2=fig_main.add_subplot(gs[1],sharex=ax1)
    ax3=fig_main.add_subplot(gs[2],sharex=ax1); ax4=fig_main.add_subplot(gs[3],sharex=ax1)
    for ax in [ax1,ax2,ax3,ax4]: estilizar_ax(ax)

    idx=df_raw.index; cs=df_raw["Close"]
    if hmm_ok:
        ca=df_raw["Close"].reindex(feat_idx)
        for i in range(best_model.n_components):
            mask=states==i
            ax1.scatter(feat_idx[mask],ca[mask],color=lmap[i]["color"],s=12,alpha=0.75,zorder=3,
                       label=lmap[i]["nombre"])
    ax1.plot(idx,cs,color="#1a2a4a",lw=0.7,alpha=0.5,zorder=2)
    ax1.plot(idx,ind["ema9"],color="#ffdd44",lw=1,alpha=0.85,label="EMA9")
    ax1.plot(idx,ind["ema21"],color="#ff8844",lw=1,alpha=0.85,label="EMA21")
    ax1.plot(idx,ind["ema50"],color="#cc44ff",lw=1.2,alpha=0.9,label="EMA50")
    ax1.fill_between(idx,ind["bb_up"],ind["bb_lo"],alpha=0.06,color="#4488ff")
    ax1.plot(idx,ind["bb_up"],color="#224488",lw=0.7,ls="--")
    ax1.plot(idx,ind["bb_lo"],color="#224488",lw=0.7,ls="--")
    ax1.legend(loc="upper left",fontsize=6,framealpha=0.3,ncol=3)
    ax1.set_ylabel("Precio",color="#2a4060",fontsize=8)
    ax1.tick_params(labelbottom=False)

    cv=["#00ff88" if df_raw["Close"].iloc[i]>=df_raw["Open"].iloc[i] else "#ff3355" for i in range(len(df_raw))]
    ax2.bar(idx,df_raw["Volume"],color=cv,alpha=0.6,width=0.8)
    ax2.plot(idx,ind["vol_sma"],color="#ffaa00",lw=1,alpha=0.8)
    ax2.set_ylabel("Vol",color="#2a4060",fontsize=7); ax2.tick_params(labelbottom=False)

    ax3.plot(idx,ind["rsi"],color="#ff8844",lw=1.2,label="RSI")
    ax3.plot(idx,ind["stoch_k"],color="#44aaff",lw=0.9,alpha=0.7,label="StochK")
    ax3.axhline(70,color="#ff3355",lw=0.7,ls="--",alpha=0.6)
    ax3.axhline(30,color="#00ff88",lw=0.7,ls="--",alpha=0.6)
    ax3.fill_between(idx,70,100,alpha=0.05,color="#ff3355")
    ax3.fill_between(idx,0,30,alpha=0.05,color="#00ff88")
    ax3.set_ylim(0,100); ax3.legend(loc="upper left",fontsize=6,framealpha=0.3)
    ax3.set_ylabel("RSI",color="#2a4060",fontsize=7); ax3.tick_params(labelbottom=False)

    ch=["#00ff88" if v>=0 else "#ff3355" for v in ind["macd_hist"]]
    ax4.bar(idx,ind["macd_hist"],color=ch,alpha=0.7,width=0.8)
    ax4.plot(idx,ind["macd"],color="#4488ff",lw=1.2,label="MACD")
    ax4.plot(idx,ind["macd_sig"],color="#ffaa00",lw=1,label="Signal")
    ax4.axhline(0,color="#1a2a4a",lw=0.5,ls=":")
    ax4.legend(loc="upper left",fontsize=6,framealpha=0.3)
    ax4.set_ylabel("MACD",color="#2a4060",fontsize=7)
    ax4.tick_params(labelrotation=25,labelsize=6)

    render_fig(fig_main)

    # ── PANEL DE SEÑAL + SCORE (debajo del gráfico) ──────────
    pa,pb,pc=st.columns([2,1,1])
    with pa:
        st.markdown(f'<div class="sbox {cls}"><div style="font-size:1.05rem;font-weight:700;margin-bottom:5px">{señal}</div><div style="opacity:0.85">{desc}</div></div>',unsafe_allow_html=True)
        if sl:
            st.markdown(f'<div class="mc"><div class="ml">Niveles sugeridos (ATR)</div><span style="color:#ff3355">🛑 SL: {fp(sl)}</span>  &nbsp;&nbsp; <span style="color:#ffaa00">🎯 TP1: {fp(tp1)}</span>  &nbsp;&nbsp; <span style="color:#00ff88">🎯 TP2: {fp(tp2)}</span></div>',unsafe_allow_html=True)
    with pb:
        if hmm_ok:
            ea=lmap[states[-1]]
            st.markdown(f'<div class="mc" style="border-left:3px solid {ea["color"]}"><div class="ml">Régimen HMM</div><div class="mv" style="color:{ea["color"]};font-size:0.95rem">{ea["emoji"]} {ea["nombre"]}</div><div style="color:#2a4060;font-size:0.7rem">Perm: {best_model.transmat_[states[-1],states[-1]]*100:.0f}%</div></div>',unsafe_allow_html=True)
    with pc:
        st.markdown(f'<div class="mc"><div class="ml">Score Trading</div><div class="mv" style="color:{sc_col}">{score}/100</div><div class="bw"><div class="bf" style="width:{score}%;background:{sc_col}"></div></div></div>',unsafe_allow_html=True)

    # Score desglose
    with st.expander("📊 Desglose del Score"):
        for nm,pts,mx,em in desglose:
            pct=pts/mx*100; col="#00ff88" if pct>=70 else("#ffaa00" if pct>=40 else "#ff3355")
            st.markdown(f'<div style="margin:3px 0"><div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#4a6080"><span>{em} {nm}</span><span style="color:{col}">{pts}/{mx}</span></div><div class="bw" style="height:6px"><div class="bf" style="width:{pct:.0f}%;background:{col}"></div></div></div>',unsafe_allow_html=True)

    st.divider()
    st.markdown("## ⚛️ ANÁLISIS CUÁNTICO")

    # ══════════════════════════════════════════════════════════
    #  MÓDULO 1
    # ══════════════════════════════════════════════════════════
    if mod_osc:
        with st.expander("🎵 MÓDULO 1 — Oscilador Armónico · Ec. de Schrödinger", expanded=True):
            st.markdown('<div class="qcard"><div class="qtitle">Ecuación de Schrödinger · Ĥψ = Eψ</div>El precio como <b>partícula cuántica</b> en un pozo parabólico. Los <b>niveles E_n = ℏω(n+½)</b> son soportes/resistencias naturales. <b>|ψ|²</b> = densidad de probabilidad del precio futuro.</div>',unsafe_allow_html=True)
            with st.spinner("Resolviendo Schrödinger..."):
                qho=quantum_harmonic_oscillator(df_raw["Close"].values)
            fig_m1=mk_fig(figsize=(14,5))
            ax_a=fig_m1.add_subplot(1,2,1); estilizar_ax(ax_a)
            ax_a.plot(df_raw["Close"].values,color="#4488ff",lw=1.2,label="Precio")
            clv=["#00ff88","#44aaff","#ffaa00","#ff8844","#ff3355"]
            pa_arr=df_raw["Close"].values
            for i,(up,dn) in enumerate(zip(qho["price_levels_up"],qho["price_levels_down"])):
                if 0<up<=pa_arr.max()*1.3: ax_a.axhline(up,color=clv[i],lw=1,ls="--",alpha=0.8,label=f"E{i+1}↑{fp(up)}")
                if 0<dn>=pa_arr.min()*0.7: ax_a.axhline(dn,color=clv[i],lw=1,ls=":",alpha=0.6)
            ax_a.legend(fontsize=6,framealpha=0.3,loc="upper left")
            ax_a.set_title("Niveles de Energía — Soportes/Resistencias",color="#4488ff",fontsize=9)
            ax_b=fig_m1.add_subplot(1,2,2); estilizar_ax(ax_b)
            ax_b.fill_betweenx(qho["x_grid"]*qho["sigma"]+qho["mu"],0,qho["psi_sq"],alpha=0.35,color="#4488ff")
            ax_b.plot(qho["psi_sq"],qho["x_grid"]*qho["sigma"]+qho["mu"],color="#00aaff",lw=2)
            ax_b.axhline(pa_arr[-1],color="#ffdd44",lw=1.5,ls="--",label=f"Precio:{fp(pa_arr[-1])}")
            ax_b.axhline(qho["mu"],color="#00ff88",lw=1,ls=":",label=f"Equil.:{fp(qho['mu'])}")
            ax_b.legend(fontsize=6,framealpha=0.3)
            ax_b.set_title("|ψ|² — Densidad de Probabilidad Futura",color="#4488ff",fontsize=9)
            ax_b.set_xlabel("Probabilidad",color="#2a4060",fontsize=8)
            plt.tight_layout()
            render_fig(fig_m1)
            # métricas en fila simple
            zcol={"REBOTE CUÁNTICO":"#00ff88","EQUILIBRIO":"#4488ff","TRANSICIÓN":"#ffaa00"}.get(qho["zona"],"#aaa")
            r1,r2,r3,r4,r5,r6=st.columns(6)
            r1.markdown(f'<div class="mc" style="border-left:2px solid {zcol}"><div class="ml">Zona</div><div class="mv" style="color:{zcol};font-size:0.82rem">{qho["zona"]}</div></div>',unsafe_allow_html=True)
            r2.markdown(f'<div class="mc"><div class="ml">ω (FFT)</div><div class="mv">{qho["omega"]:.4f}</div></div>',unsafe_allow_html=True)
            r3.markdown(f'<div class="mc"><div class="ml">V(x) potencial</div><div class="mv">{qho["V_current"]:.4f}</div></div>',unsafe_allow_html=True)
            for i,(col_r,up,dn) in enumerate(zip([r4,r5,r6],qho["price_levels_up"][:3],qho["price_levels_down"][:3])):
                col_r.markdown(f'<div class="mc"><div class="ml">E{i+1}</div><div style="color:#00ff88;font-size:0.82rem">↑{fp(up)}</div><div style="color:#ff3355;font-size:0.82rem">↓{fp(dn)}</div></div>',unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    #  MÓDULO 2
    # ══════════════════════════════════════════════════════════
    if mod_hei:
        with st.expander("🌊 MÓDULO 2 — Principio de Incertidumbre de Heisenberg", expanded=True):
            st.markdown('<div class="qcard"><div class="qtitle">ΔP · Δx ≥ ℏ/2 — Aplicado al Mercado</div>No podemos conocer precio y momentum con precisión simultánea. <b>U pequeño = señales confiables. U grande = caos.</b> El <b>túnel cuántico</b> detecta breakouts reales en zona predecible.</div>',unsafe_allow_html=True)
            with st.spinner("Calculando incertidumbre..."):
                heis=heisenberg_uncertainty(df_raw["Close"].values)
            fig_m2=mk_fig(figsize=(14,8))
            gs2=gridspec.GridSpec(3,1,figure=fig_m2,hspace=0.12)
            ax2a=fig_m2.add_subplot(gs2[0]); ax2b=fig_m2.add_subplot(gs2[1],sharex=ax2a); ax2c=fig_m2.add_subplot(gs2[2],sharex=ax2a)
            for axi in [ax2a,ax2b,ax2c]: estilizar_ax(axi)
            nh=len(df_raw["Close"].values)
            ax2a.plot(df_raw["Close"].values,color="#4488ff",lw=1.2,label="Precio")
            ti=np.where(heis["tunel"].values)[0]
            if len(ti): ax2a.scatter(ti,df_raw["Close"].values[ti],color="#ffdd44",s=40,zorder=5,label="⚡ Túnel Cuántico")
            ax2a.legend(fontsize=7,framealpha=0.3); ax2a.set_title("Precio + Tunneling Cuántico",color="#4488ff",fontsize=9)
            ax2b.fill_between(range(nh),heis["delta_x"].values,alpha=0.35,color="#ff8844")
            ax2b.plot(heis["delta_x"].values,color="#ff8844",lw=1.2); ax2b.set_title("Δx Incertidumbre de Posición",color="#ff8844",fontsize=9)
            uv=heis["U_norm"].values
            cu=["#00ff88" if not pd.isna(v) and v<0.7 else("#ffaa00" if not pd.isna(v) and v<1.3 else "#ff3355") for v in uv]
            ax2c.bar(range(nh),np.nan_to_num(uv),color=cu,alpha=0.8,width=0.8)
            ax2c.axhline(1.0,color="#ffaa00",lw=1,ls="--",alpha=0.7,label="ℏ_mercado")
            ax2c.legend(fontsize=7,framealpha=0.3); ax2c.set_title("Producto U (🟢 predecible · 🔴 caótico)",color="#4488ff",fontsize=9)
            plt.tight_layout(); render_fig(fig_m2)
            ec={"BAJA INCERTIDUMBRE":"#00ff88","INCERTIDUMBRE NORMAL":"#ffaa00","ALTA INCERTIDUMBRE":"#ff3355"}.get(heis["estado_actual"],"#aaa")
            h1,h2,h3,h4=st.columns(4)
            h1.markdown(f'<div class="mc" style="border-left:2px solid {ec}"><div class="ml">Estado Heisenberg</div><div class="mv" style="color:{ec};font-size:0.82rem">{heis["estado_actual"]}</div></div>',unsafe_allow_html=True)
            h2.markdown(f'<div class="mc"><div class="ml">Confianza Operativa</div><div class="mv">{heis["confianza"]}%</div><div class="bw"><div class="bf" style="width:{heis["confianza"]}%;background:{ec}"></div></div></div>',unsafe_allow_html=True)
            h3.markdown(f'<div class="mc"><div class="ml">Túneles (breakouts)</div><div class="mv" style="color:#ffdd44">{int(heis["tunel"].sum())}</div></div>',unsafe_allow_html=True)
            h4.markdown(f'<div class="mc"><div class="ml">ℏ Mercado</div><div class="mv">{heis["hbar_mkt"]:.6f}</div></div>',unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    #  MÓDULO 3
    # ══════════════════════════════════════════════════════════
    if mod_kal:
        with st.expander("📡 MÓDULO 3 — Filtro de Kalman (Precio Real sin Ruido)", expanded=True):
            st.markdown('<div class="qcard"><div class="qtitle">Filtro de Kalman — Ecuaciones de Riccati</div>Estima el <b>precio real</b> eliminando ruido de mercado. Estado: <b>x=[precio, velocidad]</b>. Los <b>cruces precio/Kalman</b> generan señales de entrada y salida precisas.</div>',unsafe_allow_html=True)
            with st.spinner("Ejecutando Kalman..."):
                kal=kalman_filter(df_raw["Close"].values)
            pv=df_raw["Close"].values; nk=len(pv)
            dk=pv-kal["precio_kalman"]
            ck_up=np.where((dk[1:]>0)&(dk[:-1]<=0))[0]+1
            ck_dn=np.where((dk[1:]<0)&(dk[:-1]>=0))[0]+1
            fig_m3=mk_fig(figsize=(14,7))
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
            ak1.set_title("Precio Real (Kalman) + Señales de Cruce",color="#4488ff",fontsize=9)
            vv=kal["velocidad"]
            cv_k=["#00ff88" if v>0 else "#ff3355" for v in vv]
            ak2.bar(range(nk),vv,color=cv_k,alpha=0.7,width=0.8)
            ak2.axhline(0,color="#2a4060",lw=1,ls=":")
            ak2.set_title("Velocidad d[precio]/dt (Kalman)",color="#4488ff",fontsize=9)
            plt.tight_layout(); render_fig(fig_m3)
            tc={"ALCISTA ↑":"#00ff88","BAJISTA ↓":"#ff3355","LATERAL →":"#ffaa00"}.get(kal["tendencia"],"#aaa")
            k1,k2,k3,k4=st.columns(4)
            k1.markdown(f'<div class="mc" style="border-left:2px solid {tc}"><div class="ml">Tendencia Kalman</div><div class="mv" style="color:{tc}">{kal["tendencia"]}</div></div>',unsafe_allow_html=True)
            dc=kal["diff_pct"]; dcc="#00ff88" if dc>0 else "#ff3355"
            k2.markdown(f'<div class="mc"><div class="ml">Precio vs Kalman</div><div class="mv" style="color:{dcc}">{dc:+.2f}%</div></div>',unsafe_allow_html=True)
            k3.markdown(f'<div class="mc"><div class="ml">Velocidad</div><div class="mv" style="color:{tc}">{kal["vel_actual"]:+.6f}</div></div>',unsafe_allow_html=True)
            k4.markdown(f'<div class="mc"><div class="ml">Cruces</div><div style="color:#00ff88;font-size:0.85rem">▲{len(ck_up)}</div><div style="color:#ff3355;font-size:0.85rem">▼{len(ck_dn)}</div></div>',unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    #  MÓDULO 4
    # ══════════════════════════════════════════════════════════
    if mod_ent:
        with st.expander("🔗 MÓDULO 4 — Entrelazamiento Cuántico de Activos", expanded=True):
            st.markdown('<div class="qcard"><div class="qtitle">Entropía de Von Neumann — S = −Tr(ρ ln ρ)</div>Correlación cuántica con mercados globales. <b>S≈0</b> = muy entrelazado. <b>Espejo cuántico (ρ≈−1)</b> = cobertura natural.</div>',unsafe_allow_html=True)
            with st.spinner("Calculando entrelazamiento (datos diarios)..."):
                ent=quantum_entanglement(ticker_final,fuente)
            if ent is None:
                st.info("ℹ️ No hay suficientes datos diarios para calcular correlaciones. Intenta con 1D · 1 mes o mayor.")
            else:
                fig_m4=mk_fig(figsize=(14,5))
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
                    st.markdown(f'<div class="mc"><div class="ml">Entropía Von Neumann</div><div class="mv" style="color:{svc}">{ent["S_vn"]:.3f}</div><div class="bw"><div class="bf" style="width:{svp:.0f}%;background:{svc}"></div></div><div style="color:#2a4060;font-size:0.7rem">{"Alta independencia" if svp>60 else("Entrelazado" if svp<30 else "Moderado")}</div></div>',unsafe_allow_html=True)
                with e2:
                    for nm_e,dat_e in ent["correlaciones"].items():
                        tip_e,col_e=dat_e["tipo"]
                        st.markdown(f'<div class="mc" style="padding:6px 12px;border-left:3px solid {col_e}"><div style="display:flex;justify-content:space-between;font-size:0.78rem"><span style="color:#c0d0e0">{nm_e}</span><span style="color:{col_e}">{tip_e} ρ={dat_e["corr"]:.2f}</span></div></div>',unsafe_allow_html=True)

    # ── TABLA ────────────────────────────────────────────────
    with st.expander("📊 Tabla de indicadores"):
        def fv(v,d=4): return f"{v:.{d}f}" if not pd.isna(v) else "N/D"
        t_data={"Indicador":["RSI(14)","MACD","MACD Hist","BB %B","ATR%","EMA9","EMA21","EMA50","StochRSI K","Vol Ratio","Cambio 1p","Cambio 3p"],
                "Valor":[fv(ind["rsi"].iloc[-1],1),fv(ind["macd"].iloc[-1],6),fv(ind["macd_hist"].iloc[-1],6),
                         fv(ind["bb_pct"].iloc[-1],3),f"{fv(ind['atr_pct'].iloc[-1],2)}%",
                         fv(ind["ema9"].iloc[-1],4),fv(ind["ema21"].iloc[-1],4),fv(ind["ema50"].iloc[-1],4),
                         fv(ind["stoch_k"].iloc[-1],1),f"{fv(ind['vol_ratio'].iloc[-1],2)}x",
                         f"{fv(ind['chg1'].iloc[-1],2)}%",f"{fv(ind['chg3'].iloc[-1],2)}%"]}
        st.dataframe(pd.DataFrame(t_data),use_container_width=True,hide_index=True)
