%%writefile app.py
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from hmmlearn import hmm
import requests
from datetime import datetime, timedelta
from scipy import signal as scipy_signal
from scipy.linalg import solve
import time
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI.Lino Quantum Engine", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; }
    .main { background-color: #050810; }
    .stSidebar { background-color: #08090f; border-right: 1px solid #0d1a2e; }
    h1,h2,h3 { font-family:'Orbitron',monospace !important; letter-spacing:2px; }
    .quantum-card {
        background: linear-gradient(135deg,#080c18,#0d1428);
        border: 1px solid #1a2a4a; border-radius:10px;
        padding:16px 20px; margin:8px 0;
        font-family:'Share Tech Mono',monospace;
    }
    .quantum-title {
        font-family:'Orbitron',monospace; font-size:0.85rem;
        color:#4488ff; letter-spacing:3px; text-transform:uppercase;
        margin-bottom:8px; border-bottom:1px solid #1a2a4a; padding-bottom:6px;
    }
    .signal-box { border-radius:8px; padding:14px 18px; margin:8px 0;
        font-family:'Share Tech Mono',monospace; font-size:0.82rem; border-left:4px solid; }
    .signal-compra { background:#001a0a; border-color:#00ff88; color:#00ff88; }
    .signal-venta  { background:#1a0005; border-color:#ff3355; color:#ff3355; }
    .signal-neutro { background:#0a0d1a; border-color:#4488ff; color:#7aabff; }
    .signal-espera { background:#1a1200; border-color:#ffaa00; color:#ffaa00; }
    .metric-card { background:#08090f; border:1px solid #1a2a4a;
        border-radius:8px; padding:10px 14px; margin:4px 0;
        font-family:'Share Tech Mono',monospace; }
    .metric-label { color:#2a4060; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px; }
    .metric-value { color:#c0d8ff; font-size:1.2rem; font-weight:700; }
    .score-bar-wrap { background:#0d1428; border-radius:4px; height:10px; margin:5px 0; overflow:hidden; }
    .score-bar { height:100%; border-radius:4px; }
    .stButton>button {
        background:linear-gradient(135deg,#001166,#0044cc,#0088ff);
        color:#aaddff; border:none; border-radius:6px;
        font-family:'Orbitron',monospace; font-weight:700;
        font-size:0.8rem; letter-spacing:2px;
        padding:0.7rem 1rem; width:100%; transition:all 0.3s;
    }
    .stButton>button:hover { background:linear-gradient(135deg,#0022aa,#0066ff,#00aaff);
        color:white; transform:translateY(-2px); box-shadow:0 4px 20px #0044ff44; }
    </style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  TIMEFRAMES
# ══════════════════════════════════════════════════════════════
TIMEFRAMES = {
    "1H  · 3 días"   : ("1h",  "1h",   3,   "3 días · velas 1H"),
    "4H  · 10 días"  : ("4h",  "1h",   10,  "10 días · velas 4H"),
    "1D  · 1 mes"    : ("1d",  "1d",   30,  "1 mes · velas diarias"),
    "1D  · 3 meses"  : ("1d",  "1d",   90,  "3 meses · velas diarias"),
    "1D  · 6 meses"  : ("1d",  "1d",   180, "6 meses · velas diarias"),
    "1W  · 1 año"    : ("1d",  "1wk",  365, "1 año · velas semanales"),
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
    ibn, iyf, dias, _ = TIMEFRAMES[tf_key]
    if fuente=="binance":   return binance_descargar(ticker, ibn, dias)
    elif fuente=="coingecko": return coingecko_descargar(ticker, dias)
    else:                   return yahoo_descargar(ticker, iyf, dias)

# ══════════════════════════════════════════════════════════════
#  INDICADORES TÉCNICOS CLÁSICOS
# ══════════════════════════════════════════════════════════════
def calcular_rsi(c,p=14):
    d=c.diff(); g=d.clip(lower=0).rolling(p).mean(); l=(-d.clip(upper=0)).rolling(p).mean()
    return 100-(100/(1+g/l.replace(0,np.nan)))

def calcular_macd(c,f=12,s=26,sig=9):
    ef=c.ewm(span=f,adjust=False).mean(); es=c.ewm(span=s,adjust=False).mean()
    m=ef-es; sl=m.ewm(span=sig,adjust=False).mean(); return m,sl,m-sl

def calcular_bb(c,p=20,k=2):
    sma=c.rolling(p).mean(); std=c.rolling(p).std()
    up=sma+k*std; lo=sma-k*std
    pct=(c-lo)/(up-lo+1e-9); return up,sma,lo,pct

def calcular_atr(h,l,c,p=14):
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.rolling(p).mean()

def calcular_stoch_rsi(c,p=14,sk=3,sd=3):
    rsi=calcular_rsi(c,p); mn=rsi.rolling(p).min(); mx=rsi.rolling(p).max()
    st=(rsi-mn)/(mx-mn+1e-9)*100
    k=st.rolling(sk).mean(); d=k.rolling(sd).mean(); return k,d

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
    ind["vol_ratio"]=df["Volume"]/ind["vol_sma"]
    ind["momentum"]=c.pct_change(5)*100
    ind["ret_log"]=np.log(c/c.shift(1))
    ind["vol_ann"]=ind["ret_log"].rolling(15).std()*np.sqrt(252)*100
    return ind

# ══════════════════════════════════════════════════════════════
#  HMM CLÁSICO
# ══════════════════════════════════════════════════════════════
def entrenar_hmm(df, ind):
    ret=ind["ret_log"].dropna(); vol=ind["vol_ann"].dropna(); mom=ind["momentum"].dropna()
    feat=pd.concat([ret,vol,mom],axis=1).dropna(); feat.columns=["r","v","m"]
    X=(feat-feat.mean())/feat.std()+np.random.normal(0,1e-6,feat.shape)
    best_m,best_b=None,np.inf
    for n in range(2,min(5,max(2,len(X)//20))+1):
        try:
            m=hmm.GaussianHMM(n_components=n,covariance_type="full",n_iter=2000,random_state=42)
            m.fit(X)
            b=m.bic(X)
            if b<best_b: best_b,best_m=b,m
        except: pass
    states=best_m.predict(X)
    means=best_m.means_[:,0]; idx_r=np.argsort(means); lmap={}
    lmap[idx_r[0]] ={"nombre":"PÁNICO / BAJISTA",    "color":"#ff3355","emoji":"🔴","señal":"SALIR"}
    lmap[idx_r[-1]]={"nombre":"ALCISTA / EUFORIA",   "color":"#00ff88","emoji":"🟢","señal":"VIGILAR SALIDA"}
    for i in range(best_m.n_components):
        if i not in lmap:
            if means[i]>0: lmap[i]={"nombre":"ACUMULACIÓN",        "color":"#4488ff","emoji":"🔵","señal":"POSIBLE ENTRADA"}
            else:          lmap[i]={"nombre":"LATERAL/DISTRIBUCIÓN","color":"#ffaa00","emoji":"🟡","señal":"CAUTELA"}
    return best_m,states,feat.index,lmap

# ══════════════════════════════════════════════════════════════
#  ▓▓  MÓDULO 1: OSCILADOR ARMÓNICO CUÁNTICO  ▓▓
#  Modela el precio como partícula en pozo cuántico.
#  Los niveles de energía = soportes/resistencias naturales.
#  La función de onda ψ(x) = probabilidad de precio futuro.
# ══════════════════════════════════════════════════════════════
def quantum_harmonic_oscillator(prices, n_levels=5):
    """
    Ecuación de Schrödinger para oscilador armónico:
      Hψ = Eψ  donde H = -ℏ²/2m · d²/dx² + ½mω²x²
    
    Aplicado al mercado:
      - x = desviación del precio respecto a su media (el "equilibrio")
      - ω = frecuencia natural del ciclo de precios (estimada de FFT)
      - Niveles de energía E_n = ℏω(n+½) → soporte/resistencia cuántica
      - ψ²(x) = densidad de probabilidad del precio
    """
    c = np.array(prices, dtype=float)
    N = len(c)
    
    # Normalizar precios alrededor de su media (equilibrio cuántico)
    mu    = np.mean(c)
    sigma = np.std(c)
    x_norm = (c - mu) / sigma   # posición normalizada
    
    # Estimar frecuencia natural ω via FFT (ciclo dominante)
    fft_vals = np.abs(np.fft.rfft(x_norm - np.mean(x_norm)))
    freqs    = np.fft.rfftfreq(N)
    if len(fft_vals) > 2:
        dom_freq = freqs[np.argmax(fft_vals[1:])+1]
        omega    = 2 * np.pi * max(dom_freq, 1/N)
    else:
        omega = 2 * np.pi / N
    
    # Niveles de energía cuántica: E_n = (n + 0.5) * ω  (ℏ=1)
    energy_levels = [(n + 0.5) * omega for n in range(n_levels)]
    
    # Convertir niveles de energía a precios reales
    # E = ½mω²x² → x = sqrt(2E/ω²)  → precio = mu ± x*sigma
    price_levels_up   = []
    price_levels_down = []
    for E in energy_levels:
        x_level = np.sqrt(2 * E / (omega**2 + 1e-10))
        price_levels_up.append(mu + x_level * sigma)
        price_levels_down.append(mu - x_level * sigma)
    
    # Función de onda ψ_0(x) = exp(-x²/2) · H_n(x)  (estado base)
    # Usando estado base gaussiano para la posición actual
    x_grid   = np.linspace(x_norm.min()-1, x_norm.max()+1, 500)
    psi_sq   = np.exp(-x_grid**2) / (np.sqrt(np.pi))  # |ψ₀|² estado base
    
    # Agregar excitaciones: superposición de primeros estados
    for n in range(1, 4):
        # Polinomio de Hermite H_n
        if n == 1:   Hn = 2 * x_grid
        elif n == 2: Hn = 4*x_grid**2 - 2
        elif n == 3: Hn = 8*x_grid**3 - 12*x_grid
        coeff = np.exp(-energy_levels[n] * 0.5)   # peso por energía
        psi_sq += coeff * Hn**2 * np.exp(-x_grid**2)
    
    psi_sq = psi_sq / (psi_sq.sum() + 1e-10)  # normalizar
    
    # Posición actual en el potencial
    x_current = x_norm[-1]
    V_current  = 0.5 * omega**2 * x_current**2   # energía potencial actual
    
    # ¿Está el precio en zona de rebote (pozo) o en resistencia (cresta)?
    zona = "REBOTE CUÁNTICO" if abs(x_current) > 1.5 else ("EQUILIBRIO" if abs(x_current) < 0.5 else "TRANSICIÓN")
    
    return {
        "x_grid": x_grid, "psi_sq": psi_sq,
        "price_levels_up": price_levels_up,
        "price_levels_down": price_levels_down,
        "energy_levels": energy_levels,
        "omega": omega, "mu": mu, "sigma": sigma,
        "x_current": x_current, "V_current": V_current,
        "zona": zona, "x_norm": x_norm
    }


# ══════════════════════════════════════════════════════════════
#  ▓▓  MÓDULO 2: PRINCIPIO DE INCERTIDUMBRE DE HEISENBERG  ▓▓
#  Δprecio · Δmomento ≥ ℏ/2
#  En mercados: no podemos conocer precio Y velocidad exactos.
#  Cuantifica la zona de incertidumbre operativa.
# ══════════════════════════════════════════════════════════════
def heisenberg_uncertainty(prices, ventana=20):
    """
    Adaptación financiera del principio ΔxΔp ≥ ℏ/2:
      - Δx = incertidumbre en precio (desviación estándar local)
      - Δp = incertidumbre en momentum (desviación del retorno)
      - ℏ_mercado = constante de incertidumbre empírica del activo
    
    Producto de incertidumbre: U = Δx · Δp
    Si U es GRANDE → mercado caótico, señales poco confiables
    Si U es PEQUEÑO → mercado predecible, mejores oportunidades
    """
    c = pd.Series(prices)
    
    # Δx: incertidumbre de posición = volatilidad de precio normalizada
    delta_x = c.rolling(ventana).std() / c.rolling(ventana).mean()
    
    # Δp: incertidumbre de momentum = volatilidad del retorno
    ret     = np.log(c / c.shift(1))
    delta_p = ret.rolling(ventana).std() * np.sqrt(ventana)
    
    # Producto de incertidumbre U = Δx · Δp (análogo a ΔxΔp)
    U = delta_x * delta_p
    
    # ℏ_mercado: constante de Planck del mercado = mediana histórica de U
    hbar_mkt = U.median()
    
    # Zona de confianza operativa
    U_norm = U / (hbar_mkt + 1e-10)   # relativo a la constante histórica
    
    # Clasificación cuántica
    clasificacion = pd.Series(index=c.index, dtype=str)
    clasificacion[U_norm < 0.7]  = "BAJA INCERTIDUMBRE"
    clasificacion[(U_norm >= 0.7) & (U_norm < 1.3)] = "INCERTIDUMBRE NORMAL"
    clasificacion[U_norm >= 1.3] = "ALTA INCERTIDUMBRE"
    clasificacion.fillna("INCERTIDUMBRE NORMAL", inplace=True)
    
    # Tunel cuántico: detectar rupturas estadísticas inesperadas
    # Si el precio cruza 2σ pero U es bajo → posible túnel cuántico (breakout real)
    zscore = (c - c.rolling(ventana).mean()) / (c.rolling(ventana).std() + 1e-10)
    tunel  = (zscore.abs() > 2.0) & (U_norm < 1.0)
    
    estado_actual = clasificacion.iloc[-1]
    U_actual      = U_norm.iloc[-1]
    confianza     = max(0, min(100, int((1.5 - min(U_actual, 1.5)) / 1.5 * 100)))
    
    return {
        "delta_x": delta_x, "delta_p": delta_p,
        "U": U, "U_norm": U_norm, "hbar_mkt": hbar_mkt,
        "clasificacion": clasificacion, "tunel": tunel,
        "estado_actual": estado_actual, "U_actual": U_actual,
        "confianza": confianza, "zscore": zscore
    }


# ══════════════════════════════════════════════════════════════
#  ▓▓  MÓDULO 3: FILTRO DE KALMAN  ▓▓
#  Estimación óptima del estado "verdadero" del precio
#  eliminando ruido de mercado con ecuaciones de Riccati.
# ══════════════════════════════════════════════════════════════
def kalman_filter(prices):
    """
    Filtro de Kalman (hermano cuántico del HMM):
    
    Modelo de estado:
      x_t = A·x_{t-1} + w_t    (ecuación de transición)  w ~ N(0,Q)
      z_t = H·x_t   + v_t      (ecuación de observación)  v ~ N(0,R)
    
    Estado: x = [precio_real, velocidad_precio]
    Observación: z = precio_observado (con ruido de mercado)
    
    Las ecuaciones de Kalman dan la estimación ÓPTIMA del precio real
    minimizando el error cuadrático medio.
    """
    z = np.array(prices, dtype=float)
    N = len(z)
    
    # Matrices del sistema
    dt = 1.0
    A  = np.array([[1, dt],   # transición: precio += velocidad
                   [0,  1]])  # velocidad constante (modelo CV)
    H  = np.array([[1, 0]])   # observamos solo el precio
    
    # Ruidos — estimados de los datos
    sigma_obs    = np.std(np.diff(z)) * 0.5    # ruido de observación
    sigma_proc   = np.std(np.diff(z)) * 0.1    # ruido de proceso
    R = np.array([[sigma_obs**2]])             # covarianza de observación
    Q = np.array([[sigma_proc**2, 0],
                  [0, (sigma_proc*0.1)**2]])   # covarianza de proceso
    
    # Inicialización
    x_est  = np.zeros((N, 2))    # estado estimado [precio, velocidad]
    P_list = [np.eye(2) * sigma_obs**2]
    
    x_est[0] = [z[0], 0]
    
    # Loop de Kalman
    for t in range(1, N):
        # PREDICCIÓN
        x_pred = A @ x_est[t-1]
        P_pred = A @ P_list[-1] @ A.T + Q
        
        # GANANCIA DE KALMAN
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        # ACTUALIZACIÓN
        innovation    = z[t] - H @ x_pred
        x_est[t]      = x_pred + K.flatten() * innovation.flatten()
        P_updated     = (np.eye(2) - K @ H) @ P_pred
        P_list.append(P_updated)
    
    precio_kalman   = x_est[:, 0]
    velocidad_kalman = x_est[:, 1]
    
    # Banda de confianza ±2σ del filtro
    varianzas  = np.array([P[0,0] for P in P_list])
    banda_sup  = precio_kalman + 2 * np.sqrt(np.abs(varianzas))
    banda_inf  = precio_kalman - 2 * np.sqrt(np.abs(varianzas))
    
    # Señal de Kalman: precio vs línea filtrada
    precio_actual   = z[-1]
    kalman_actual   = precio_kalman[-1]
    vel_actual      = velocidad_kalman[-1]
    
    # Tendencia Kalman
    if vel_actual > sigma_proc * 0.5:   tendencia_k = "ALCISTA ↑"
    elif vel_actual < -sigma_proc * 0.5: tendencia_k = "BAJISTA ↓"
    else:                                tendencia_k = "LATERAL →"
    
    # Precio sobre/bajo filtro = señal
    diff_pct = (precio_actual - kalman_actual) / (kalman_actual + 1e-10) * 100
    
    return {
        "precio_kalman": precio_kalman,
        "velocidad": velocidad_kalman,
        "banda_sup": banda_sup, "banda_inf": banda_inf,
        "varianzas": varianzas,
        "tendencia": tendencia_k,
        "diff_pct": diff_pct,
        "vel_actual": vel_actual,
        "kalman_actual": kalman_actual
    }


# ══════════════════════════════════════════════════════════════
#  ▓▓  MÓDULO 4: ENTRELAZAMIENTO DE ACTIVOS  ▓▓
#  Correlación cuántica: ρ y entropía de Von Neumann
#  Detecta qué activos se mueven juntos o en espejo.
# ══════════════════════════════════════════════════════════════
def quantum_entanglement(df_main, ticker_main, n_dias=90):
    """
    Entrelazamiento cuántico financiero:
    
    1. Matriz de densidad ρ_ij = correlación normalizada (análogo cuántico)
    2. Entropía de Von Neumann: S = -Tr(ρ ln ρ)
       S≈0 → sistema puro (alta correlación / entrelazamiento fuerte)
       S grande → sistema mixto (activos independientes)
    3. Estado de Bell financiero: pares que se "miden" juntos
    """
    # Activos de referencia para correlacionar
    pares_ref = {
        "BTC-USD":  "₿ Bitcoin",
        "ETH-USD":  "Ξ Ethereum",
        "SPY":      "📊 S&P 500",
        "GLD":      "🥇 Oro",
        "DX-Y.NYB": "💵 USD Index",
        "^VIX":     "😰 VIX",
    }
    
    ret_main = np.log(df_main["Close"] / df_main["Close"].shift(1)).dropna()
    
    correlaciones = {}
    for sym, nombre in pares_ref.items():
        if sym == ticker_main: continue
        try:
            t   = yf.Ticker(sym)
            df2 = t.history(period="3mo", interval="1d")
            if df2.empty or len(df2) < 20: continue
            ret2 = np.log(df2["Close"] / df2["Close"].shift(1)).dropna()
            # Alinear índices
            idx  = ret_main.index.intersection(ret2.index)
            if len(idx) < 15: continue
            r1, r2 = ret_main.loc[idx].values, ret2.loc[idx].values
            corr   = np.corrcoef(r1, r2)[0,1]
            if not np.isnan(corr):
                correlaciones[nombre] = {"sym": sym, "corr": corr}
        except: pass
    
    if len(correlaciones) < 2:
        return None
    
    nombres = list(correlaciones.keys())
    n       = len(nombres)
    
    # Matriz de densidad (correlaciones normalizadas)
    C = np.zeros((n, n))
    vals = [correlaciones[nm]["corr"] for nm in nombres]
    for i in range(n):
        for j in range(n):
            C[i,j] = (vals[i] * vals[j] + 1) / 2   # normalizar [0,1]
    np.fill_diagonal(C, 1.0)
    
    # Entropía de Von Neumann: S = -Tr(ρ ln ρ)
    eigenvals = np.linalg.eigvalsh(C)
    eigenvals = eigenvals[eigenvals > 1e-10] / eigenvals.sum()
    S_vn      = -np.sum(eigenvals * np.log(eigenvals + 1e-10))
    S_max     = np.log(n)   # entropía máxima
    S_norm    = S_vn / (S_max + 1e-10)   # 0=entrelazado, 1=independiente
    
    # Clasificar pares
    for nm in nombres:
        c = correlaciones[nm]["corr"]
        if   c >  0.7: correlaciones[nm]["tipo"] = ("ENTRELAZADO ↑↑", "#00ff88")
        elif c >  0.4: correlaciones[nm]["tipo"] = ("CORRELADO ↑",    "#44aaff")
        elif c >  -0.4: correlaciones[nm]["tipo"] = ("INDEPENDIENTE ⊥","#aaaaaa")
        elif c > -0.7: correlaciones[nm]["tipo"] = ("ANTI-CORR ↑↓",  "#ffaa00")
        else:          correlaciones[nm]["tipo"] = ("ESPEJO CUÁNTICO ↓↑","#ff3355")
    
    return {
        "correlaciones": correlaciones,
        "nombres": nombres,
        "C_matrix": C,
        "S_vn": S_vn, "S_norm": S_norm,
        "eigenvals_norm": eigenvals
    }


# ══════════════════════════════════════════════════════════════
#  SCORE COMPUESTO
# ══════════════════════════════════════════════════════════════
def calcular_score(ind):
    p=[]
    rsi=ind["rsi"].iloc[-1] if not pd.isna(ind["rsi"].iloc[-1]) else 50
    if rsi<30:   p.append(("RSI Sobreventa",25,25,"🟢"))
    elif rsi<45: p.append(("RSI Zona baja",18,25,"🟡"))
    elif rsi<55: p.append(("RSI Neutral",12,25,"⚪"))
    elif rsi<70: p.append(("RSI Zona alta",7,25,"🟡"))
    else:        p.append(("RSI Sobrecompra",2,25,"🔴"))

    hist=ind["macd_hist"].iloc[-1]; prev=ind["macd_hist"].iloc[-2] if len(ind["macd_hist"])>1 else hist
    if pd.isna(hist): hist=0; prev=0
    if prev<=0 and hist>0:         p.append(("MACD Cruce alcista",20,20,"🟢"))
    elif hist>0 and hist>prev:     p.append(("MACD Acelerando ↑",15,20,"🟢"))
    elif hist>0:                   p.append(("MACD Positivo",10,20,"🟡"))
    elif prev>=0 and hist<0:       p.append(("MACD Cruce bajista",0,20,"🔴"))
    else:                          p.append(("MACD Negativo",4,20,"🔴"))

    pct=ind["bb_pct"].iloc[-1] if not pd.isna(ind["bb_pct"].iloc[-1]) else 0.5
    if pct<0.05:   p.append(("BB Banda inf",15,15,"🟢"))
    elif pct<0.35: p.append(("BB Zona baja",11,15,"🟡"))
    elif pct<0.65: p.append(("BB Centro",7,15,"⚪"))
    elif pct<0.95: p.append(("BB Zona alta",4,15,"🟡"))
    else:          p.append(("BB Banda sup",1,15,"🔴"))

    e9=ind["ema9"].iloc[-1]; e21=ind["ema21"].iloc[-1]; e50=ind["ema50"].iloc[-1]
    es=0
    if not any(pd.isna(v) for v in [e9,e21,e50]):
        if e9>e21>e50: es=20
        elif e9>e21:   es=13
        elif e9>e50:   es=8
        else:          es=3
    p.append(("EMAs alineadas",es,20,"🟢" if es>=13 else("🟡" if es>=8 else "🔴")))

    vr=ind["vol_ratio"].iloc[-1] if not pd.isna(ind["vol_ratio"].iloc[-1]) else 1
    if vr>2:   p.append(("Volumen x2+",10,10,"🟢"))
    elif vr>1.3: p.append(("Vol elevado",7,10,"🟡"))
    elif vr>0.8: p.append(("Vol normal",5,10,"⚪"))
    else:        p.append(("Vol bajo",2,10,"🔴"))

    sk=ind["stoch_k"].iloc[-1]; sd=ind["stoch_d"].iloc[-1]
    if not any(pd.isna(v) for v in [sk,sd]):
        if sk<20 and sk>sd:   p.append(("StochRSI alcista",10,10,"🟢"))
        elif sk<30:           p.append(("StochRSI S.venta",7,10,"🟡"))
        elif sk>80 and sk<sd: p.append(("StochRSI bajista",1,10,"🔴"))
        elif sk>70:           p.append(("StochRSI S.compra",3,10,"🟡"))
        else:                 p.append(("StochRSI Neutral",5,10,"⚪"))
    else: p.append(("StochRSI N/D",5,10,"⚪"))

    sc=round(sum(x[1] for x in p)/sum(x[2] for x in p)*100)
    return sc,p

def generar_señal(score, ind):
    precio=ind["bb_mid"].iloc[-1]; atr=ind["atr"].iloc[-1]
    if pd.isna(atr): atr=precio*0.02
    if score>=72:
        return("🟢 COMPRA / LONG","signal-compra",
               precio-1.5*atr, precio+2*atr, precio+3.5*atr,
               "Múltiples indicadores alcistas. Considera entrada con SL ajustado.")
    elif score>=55:
        return("🟡 VIGILAR — Posible entrada","signal-espera",
               precio-2*atr, precio+1.5*atr, precio+2.5*atr,
               "Señales mixtas, sesgo alcista. Espera confirmación.")
    elif score>=38:
        return("⚪ NEUTRAL","signal-neutro",None,None,None,
               "Sin dirección clara. Mejor esperar.")
    else:
        return("🔴 VENTA / SALIR","signal-venta",
               precio+1.5*atr, precio-2*atr, precio-3.5*atr,
               "Debilidad generalizada. Considera reducir exposición.")


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
st.sidebar.markdown("# 🌌 AI.LINO")
st.sidebar.markdown("**QUANTUM TRADING ENGINE**")
st.sidebar.divider()

query=st.sidebar.text_input("🔍 Buscar activo:",placeholder="Monad, BTC, Tesla...",key="sq")
ticker_final=None; ticker_nombre=None; fuente=None

if query and len(query)>=2:
    with st.sidebar:
        with st.spinner("Escaneando 3 fuentes..."):
            try: res_yf=yf.Search(query,max_results=3,enable_fuzzy_query=True); qyf=res_yf.quotes
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
            omap[lbl]=(cid,f"{nm} ({sym})","coingecko")
        if opts:
            sel=st.radio("",opts,label_visibility="collapsed")
            if sel: ticker_final,ticker_nombre,fuente=omap[sel]; st.success(f"✅ {ticker_final}")
        else: st.warning("Sin resultados.")
else:
    favs={"MON · Monad 🦎":("monad","Monad (MON)","coingecko"),
          "BTC/USDT 🟡":("BTCUSDT","Bitcoin/USDT","binance"),
          "ETH/USDT":("ETHUSDT","Ethereum/USDT","binance"),
          "SOL/USDT ☀️":("SOLUSDT","Solana/USDT","binance"),
          "PEPE/USDT 🐸":("PEPEUSDT","Pepe/USDT","binance"),
          "DOGE/USDT 🐕":("DOGEUSDT","Dogecoin/USDT","binance"),
          "NVDA 🟢":("NVDA","NVIDIA","yahoo"),
          "AAPL 🍎":("AAPL","Apple","yahoo"),
          "TSLA ⚡":("TSLA","Tesla","yahoo"),
          "SPY 📊":("SPY","S&P500 ETF","yahoo")}
    fs=st.sidebar.selectbox("⭐ Favoritos:",list(favs.keys()),label_visibility="collapsed")
    ticker_final,ticker_nombre,fuente=favs[fs]

st.sidebar.divider()
tf_key=st.sidebar.selectbox("⏱ Timeframe",list(TIMEFRAMES.keys()),index=2)

# Selector de módulos cuánticos
st.sidebar.markdown("**⚛️ Módulos Cuánticos:**")
mod_osc    = st.sidebar.toggle("1· Oscilador Armónico",   value=True)
mod_heisen = st.sidebar.toggle("2· Incertidumbre Heisenberg", value=True)
mod_kalman = st.sidebar.toggle("3· Filtro de Kalman",     value=True)
mod_entang = st.sidebar.toggle("4· Entrelazamiento",      value=True)

if ticker_final:
    bdg={"yahoo":"📈 Yahoo","binance":"🟡 Binance","coingecko":"🦎 CoinGecko"}.get(fuente,"")
    st.sidebar.info(f"📌 **{ticker_final}** · {bdg}")

ejecutar=st.sidebar.button("⚛️ EJECUTAR ANÁLISIS CUÁNTICO",use_container_width=True)
st.sidebar.divider()
st.sidebar.caption("⚠️ Solo educativo. No es asesoría financiera.")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
if ejecutar:
    if not ticker_final: st.error("Selecciona un instrumento."); st.stop()

    bdg={"yahoo":"📈 Yahoo Finance","binance":"🟡 Binance","coingecko":"🦎 CoinGecko"}.get(fuente,"")
    _,_,dias,tf_disp=TIMEFRAMES[tf_key]

    with st.spinner("⬇️ Descargando datos..."):
        df_raw=cargar_datos(ticker_final,fuente,tf_key)
    if df_raw is None or df_raw.empty or len(df_raw)<20:
        st.error("❌ Datos insuficientes. Prueba un timeframe mayor."); st.stop()

    with st.spinner("🧮 Calculando indicadores..."):
        ind=calcular_indicadores(df_raw)

    with st.spinner("🤖 Entrenando HMM..."):
        try: best_model,states,feat_idx,lmap=entrenar_hmm(df_raw,ind); hmm_ok=True
        except: hmm_ok=False; lmap={}

    score,desglose=calcular_score(ind)
    señal,cls,sl,tp1,tp2,desc=generar_señal(score,ind)
    precio_actual=df_raw["Close"].iloc[-1]
    cambio_1p=df_raw["Close"].pct_change(1).iloc[-1]*100

    # ── HEADER ──────────────────────────────────────────────
    st.markdown(f"## ⚛️ {ticker_nombre}")
    st.caption(f"{bdg} · {tf_disp} · {len(df_raw)} velas · {datetime.utcnow().strftime('%H:%M UTC')}")

    pfmt=f"${precio_actual:,.6f}" if precio_actual<1 else f"${precio_actual:,.4f}" if precio_actual<10 else f"${precio_actual:,.2f}"
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("Precio",pfmt,f"{cambio_1p:+.2f}%")
    c2.metric("RSI(14)",f"{ind['rsi'].iloc[-1]:.1f}" if not pd.isna(ind['rsi'].iloc[-1]) else "N/D")
    c3.metric("Volatilidad",f"{ind['atr_pct'].iloc[-1]:.2f}%" if not pd.isna(ind['atr_pct'].iloc[-1]) else "N/D")
    c4.metric("Vol Ratio",f"{ind['vol_ratio'].iloc[-1]:.2f}x" if not pd.isna(ind['vol_ratio'].iloc[-1]) else "N/D")
    c5.metric("Score IA",f"{score}/100")
    st.divider()

    # ── GRÁFICO PRINCIPAL + SEÑAL ────────────────────────────
    col_ch,col_pn=st.columns([3,1])
    with col_ch:
        plt.style.use("dark_background")
        fig=plt.figure(figsize=(13,10),facecolor="#050810")
        gs=gridspec.GridSpec(4,1,figure=fig,hspace=0.06,height_ratios=[3,1,1,1])
        ax1=fig.add_subplot(gs[0]); ax2=fig.add_subplot(gs[1],sharex=ax1)
        ax3=fig.add_subplot(gs[2],sharex=ax1); ax4=fig.add_subplot(gs[3],sharex=ax1)
        idx=df_raw.index; c_ser=df_raw["Close"]
        BG="#050810"
        if hmm_ok:
            ca=df_raw["Close"].loc[feat_idx]
            for i in range(best_model.n_components):
                mask=states==i
                ax1.scatter(feat_idx[mask],ca[mask],color=lmap[i]["color"],s=10,alpha=0.7,zorder=3)
        ax1.plot(idx,c_ser,color="#1a2a4a",lw=0.8,alpha=0.5,zorder=2)
        ax1.plot(idx,ind["ema9"],color="#ffdd44",lw=1,alpha=0.8,label="EMA9")
        ax1.plot(idx,ind["ema21"],color="#ff8844",lw=1,alpha=0.8,label="EMA21")
        ax1.plot(idx,ind["ema50"],color="#cc44ff",lw=1.2,alpha=0.9,label="EMA50")
        ax1.fill_between(idx,ind["bb_up"],ind["bb_lo"],alpha=0.06,color="#4488ff")
        ax1.plot(idx,ind["bb_up"],color="#224488",lw=0.7,ls="--")
        ax1.plot(idx,ind["bb_lo"],color="#224488",lw=0.7,ls="--")
        ax1.legend(loc="upper left",fontsize=7,framealpha=0.3)
        for ax in [ax1,ax2,ax3,ax4]: ax.set_facecolor(BG); [sp.set_color("#0d1a2e") for sp in ax.spines.values()]
        ax1.tick_params(labelbottom=False,colors="#2a4060")
        cv=["#00ff88" if df_raw["Close"].iloc[i]>=df_raw["Open"].iloc[i] else "#ff3355" for i in range(len(df_raw))]
        ax2.bar(idx,df_raw["Volume"],color=cv,alpha=0.6,width=0.8)
        ax2.plot(idx,ind["vol_sma"],color="#ffaa00",lw=1,alpha=0.8)
        ax2.tick_params(labelbottom=False,colors="#2a4060"); ax2.set_ylabel("Vol",color="#2a4060",fontsize=8)
        ax3.plot(idx,ind["rsi"],color="#ff8844",lw=1.2,label="RSI")
        ax3.plot(idx,ind["stoch_k"],color="#44aaff",lw=0.9,alpha=0.7,label="StochK")
        ax3.axhline(70,color="#ff3355",lw=0.7,ls="--",alpha=0.6)
        ax3.axhline(30,color="#00ff88",lw=0.7,ls="--",alpha=0.6)
        ax3.fill_between(idx,70,100,alpha=0.05,color="#ff3355")
        ax3.fill_between(idx,0,30,alpha=0.05,color="#00ff88")
        ax3.set_ylim(0,100); ax3.legend(loc="upper left",fontsize=7,framealpha=0.3)
        ax3.tick_params(labelbottom=False,colors="#2a4060"); ax3.set_ylabel("RSI",color="#2a4060",fontsize=8)
        ch=["#00ff88" if v>=0 else "#ff3355" for v in ind["macd_hist"]]
        ax4.bar(idx,ind["macd_hist"],color=ch,alpha=0.7,width=0.8)
        ax4.plot(idx,ind["macd"],color="#4488ff",lw=1.2,label="MACD")
        ax4.plot(idx,ind["macd_sig"],color="#ffaa00",lw=1,label="Signal")
        ax4.axhline(0,color="#1a2a4a",lw=0.5,ls=":")
        ax4.legend(loc="upper left",fontsize=7,framealpha=0.3)
        ax4.tick_params(colors="#2a4060",labelrotation=30,labelsize=7); ax4.set_ylabel("MACD",color="#2a4060",fontsize=8)
        st.pyplot(fig); plt.close(fig)

    with col_pn:
        st.markdown(f'<div class="signal-box {cls}"><div style="font-size:1.05rem;font-weight:700;margin-bottom:6px">{señal}</div><div style="font-size:0.77rem;opacity:0.8">{desc}</div></div>',unsafe_allow_html=True)
        if hmm_ok:
            ea=lmap[states[-1]]
            st.markdown(f'<div class="metric-card" style="border-left:3px solid {ea["color"]}"><div class="metric-label">Régimen HMM</div><div class="metric-value" style="color:{ea["color"]};font-size:0.95rem">{ea["emoji"]} {ea["nombre"]}</div><div style="color:#2a4060;font-size:0.72rem;margin-top:3px">Permanencia: {best_model.transmat_[states[-1],states[-1]]*100:.0f}%</div></div>',unsafe_allow_html=True)
        sc=("#00ff88" if score>=65 else("#ffaa00" if score>=45 else "#ff3355"))
        st.markdown(f'<div class="metric-card"><div class="metric-label">Score Trading</div><div class="metric-value" style="color:{sc}">{score}/100</div><div class="score-bar-wrap"><div class="score-bar" style="width:{score}%;background:{sc}"></div></div></div>',unsafe_allow_html=True)
        def fp(v):
            if v is None: return "—"
            return f"${v:,.6f}" if v<1 else f"${v:,.4f}" if v<10 else f"${v:,.2f}"
        if sl:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Niveles ATR</div><div style="color:#ff3355;font-size:0.82rem">🛑 SL: {fp(sl)}</div><div style="color:#ffaa00;font-size:0.82rem">🎯 TP1: {fp(tp1)}</div><div style="color:#00ff88;font-size:0.82rem">🎯 TP2: {fp(tp2)}</div></div>',unsafe_allow_html=True)
        st.markdown("**Score:**")
        for nm,pts,mx,em in desglose:
            pct=pts/mx*100; col="#00ff88" if pct>=70 else("#ffaa00" if pct>=40 else "#ff3355")
            st.markdown(f'<div style="margin:2px 0"><div style="display:flex;justify-content:space-between;font-size:0.72rem;color:#4a6080"><span>{em} {nm}</span><span style="color:{col}">{pts}/{mx}</span></div><div class="score-bar-wrap" style="height:5px"><div class="score-bar" style="width:{pct:.0f}%;background:{col}"></div></div></div>',unsafe_allow_html=True)

    st.divider()

    # ══════════════════════════════════════════════════════════
    #  MÓDULOS CUÁNTICOS
    # ══════════════════════════════════════════════════════════
    st.markdown("## ⚛️ ANÁLISIS CUÁNTICO")

    prices_arr = df_raw["Close"].values

    # ── MÓDULO 1: OSCILADOR ARMÓNICO ────────────────────────
    if mod_osc:
        with st.expander("🎵  MÓDULO 1 — Oscilador Armónico Cuántico  (Ec. de Schrödinger)", expanded=True):
            st.markdown("""
            <div class="quantum-card">
            <div class="quantum-title">Ecuación de Schrödinger · Ĥψ = Eψ</div>
            Modela el precio como una <b>partícula cuántica en un pozo de potencial parabólico</b>.
            Los <b>niveles de energía E_n = ℏω(n+½)</b> definen soportes y resistencias naturales del activo.
            La función de onda <b>ψ²(x)</b> muestra la <i>densidad de probabilidad</i> de precio futuro.
            </div>""", unsafe_allow_html=True)

            with st.spinner("Resolviendo ecuación de Schrödinger..."):
                qho = quantum_harmonic_oscillator(prices_arr)

            # Gráfico ancho (sin st.columns alrededor — evita bug NULL)
            fig_m1, axes_m1 = plt.subplots(1, 2, figsize=(14, 5), facecolor="#050810")
            ax_m1a = axes_m1[0]; ax_m1a.set_facecolor("#050810")
            ax_m1a.plot(prices_arr, color="#4488ff", lw=1.2, alpha=0.9, label="Precio")
            clev = ["#00ff88","#44aaff","#ffaa00","#ff8844","#ff3355"]
            for i,(up,down) in enumerate(zip(qho["price_levels_up"], qho["price_levels_down"])):
                if up <= prices_arr.max()*1.2 and up > 0:
                    ax_m1a.axhline(up,   color=clev[i], lw=1, ls="--", alpha=0.8, label=f"E{i+1}↑ {fp(up)}")
                if down > 0 and down >= prices_arr.min()*0.8:
                    ax_m1a.axhline(down, color=clev[i], lw=1, ls=":",  alpha=0.6, label=f"E{i+1}↓ {fp(down)}")
            ax_m1a.legend(fontsize=7, framealpha=0.3, loc="upper left")
            ax_m1a.set_title("Niveles de Energía Cuántica — Soportes/Resistencias", color="#4488ff", fontsize=10)
            ax_m1a.tick_params(colors="#2a4060"); [sp.set_color("#0d1a2e") for sp in ax_m1a.spines.values()]
            ax_m1b = axes_m1[1]; ax_m1b.set_facecolor("#050810")
            ax_m1b.fill_betweenx(qho["x_grid"]*qho["sigma"]+qho["mu"], 0, qho["psi_sq"], alpha=0.4, color="#4488ff")
            ax_m1b.plot(qho["psi_sq"], qho["x_grid"]*qho["sigma"]+qho["mu"], color="#00aaff", lw=2)
            ax_m1b.axhline(prices_arr[-1], color="#ffdd44", lw=1.5, ls="--", label=f"Precio: {fp(prices_arr[-1])}")
            ax_m1b.axhline(qho["mu"],      color="#00ff88", lw=1,   ls=":",  label=f"Equilibrio: {fp(qho['mu'])}")
            ax_m1b.legend(fontsize=7, framealpha=0.3)
            ax_m1b.set_title("|ψ|² — Densidad de Probabilidad de Precio Futuro", color="#4488ff", fontsize=10)
            ax_m1b.set_xlabel("Probabilidad", color="#2a4060", fontsize=8)
            ax_m1b.tick_params(colors="#2a4060"); [sp.set_color("#0d1a2e") for sp in ax_m1b.spines.values()]
            plt.tight_layout()
            st.pyplot(fig_m1); plt.close(fig_m1)
            # Métricas en fila debajo
            zona_color = {"REBOTE CUÁNTICO":"#00ff88","EQUILIBRIO":"#4488ff","TRANSICIÓN":"#ffaa00"}.get(qho["zona"],"#aaaaaa")
            mq1,mq2,mq3,mq4,mq5,mq6 = st.columns(6)
            mq1.markdown(f'<div class="metric-card" style="border-left:3px solid {zona_color}"><div class="metric-label">Zona Cuántica</div><div class="metric-value" style="color:{zona_color};font-size:0.88rem">{qho["zona"]}</div></div>', unsafe_allow_html=True)
            mq2.markdown(f'<div class="metric-card"><div class="metric-label">ω (FFT)</div><div class="metric-value">{qho["omega"]:.4f}</div><div style="color:#2a4060;font-size:0.7rem">rad/período</div></div>', unsafe_allow_html=True)
            mq3.markdown(f'<div class="metric-card"><div class="metric-label">Energía V(x)</div><div class="metric-value">{qho["V_current"]:.4f}</div><div style="color:#2a4060;font-size:0.7rem">½ω²x²</div></div>', unsafe_allow_html=True)
            nivel_cols = [mq4, mq5, mq6]
            for i,(up,down) in enumerate(zip(qho["price_levels_up"][:3],qho["price_levels_down"][:3])):
                nivel_cols[i].markdown(f'<div class="metric-card"><div class="metric-label">E{i+1}</div><div style="color:#00ff88;font-size:0.82rem">↑ {fp(up)}</div><div style="color:#ff3355;font-size:0.82rem">↓ {fp(down)}</div></div>', unsafe_allow_html=True)

    # ── MÓDULO 2: HEISENBERG ─────────────────────────────────
    if mod_heisen:
        with st.expander("🌊  MÓDULO 2 — Principio de Incertidumbre de Heisenberg", expanded=True):
            st.markdown("""
            <div class="quantum-card">
            <div class="quantum-title">ΔP · Δx ≥ ℏ/2 — Aplicado al Mercado</div>
            No podemos conocer con precisión simultánea el <b>precio</b> y su <b>momentum</b>.
            <br>El <b>producto de incertidumbre U = Δx·Δp</b> cuantifica qué tan predecible es el mercado.
            <br>U pequeño = señales más confiables. U grande = mercado caótico.
            <br>Detección de <b>túnel cuántico</b>: rupturas estadísticas en zona de baja incertidumbre = breakout real.
            </div>""", unsafe_allow_html=True)

            with st.spinner("Calculando principio de incertidumbre..."):
                heis = heisenberg_uncertainty(prices_arr)

            # Gráfico ancho M2 (sin columns)
            fig_m2, axes_m2 = plt.subplots(3, 1, figsize=(13, 8), sharex=True, facecolor="#050810")
            fig_m2.subplots_adjust(hspace=0.12)
            n_h = len(prices_arr)
            axes_m2[0].plot(prices_arr, color="#4488ff", lw=1.2, label="Precio")
            tunel_idx = np.where(heis["tunel"].values)[0]
            if len(tunel_idx):
                axes_m2[0].scatter(tunel_idx, prices_arr[tunel_idx], color="#ffdd44", s=40, zorder=5, label="⚡ Túnel Cuántico")
            axes_m2[0].legend(fontsize=7, framealpha=0.3)
            axes_m2[0].set_title("Precio + Breakouts por Túnel Cuántico", color="#4488ff", fontsize=9)
            axes_m2[1].fill_between(range(n_h), heis["delta_x"].values, alpha=0.4, color="#ff8844")
            axes_m2[1].plot(heis["delta_x"].values, color="#ff8844", lw=1.2, label="Δx")
            axes_m2[1].set_title("Incertidumbre de Posición Δx", color="#ff8844", fontsize=9)
            u_vals = heis["U_norm"].values
            cu = ["#00ff88" if not pd.isna(v) and v<0.7 else ("#ffaa00" if not pd.isna(v) and v<1.3 else "#ff3355") for v in u_vals]
            axes_m2[2].bar(range(n_h), np.nan_to_num(u_vals), color=cu, alpha=0.8, width=0.8)
            axes_m2[2].axhline(1.0, color="#ffaa00", lw=1, ls="--", alpha=0.7, label="ℏ_mercado")
            axes_m2[2].legend(fontsize=7, framealpha=0.3)
            axes_m2[2].set_title("Producto U (verde=predecible, rojo=caótico)", color="#4488ff", fontsize=9)
            for ax_h in axes_m2:
                ax_h.set_facecolor("#050810"); ax_h.tick_params(colors="#2a4060")
                [sp.set_color("#0d1a2e") for sp in ax_h.spines.values()]
            st.pyplot(fig_m2); plt.close(fig_m2)
            # Métricas en fila
            e_col = {"BAJA INCERTIDUMBRE":"#00ff88","INCERTIDUMBRE NORMAL":"#ffaa00","ALTA INCERTIDUMBRE":"#ff3355"}.get(heis["estado_actual"],"#aaa")
            mh1,mh2,mh3,mh4 = st.columns(4)
            mh1.markdown(f'<div class="metric-card" style="border-left:3px solid {e_col}"><div class="metric-label">Estado Heisenberg</div><div class="metric-value" style="color:{e_col};font-size:0.85rem">{heis["estado_actual"]}</div></div>', unsafe_allow_html=True)
            mh2.markdown(f'<div class="metric-card"><div class="metric-label">Confianza Operativa</div><div class="metric-value">{heis["confianza"]}%</div><div class="score-bar-wrap"><div class="score-bar" style="width:{heis["confianza"]}%;background:{e_col}"></div></div></div>', unsafe_allow_html=True)
            mh3.markdown(f'<div class="metric-card"><div class="metric-label">Túneles Cuánticos</div><div class="metric-value" style="color:#ffdd44">{int(heis["tunel"].sum())}</div><div style="color:#2a4060;font-size:0.7rem">Breakouts reales</div></div>', unsafe_allow_html=True)
            mh4.markdown(f'<div class="metric-card"><div class="metric-label">ℏ Mercado</div><div class="metric-value">{heis["hbar_mkt"]:.6f}</div><div style="color:#2a4060;font-size:0.7rem">Constante empírica</div></div>', unsafe_allow_html=True)
    # ── MÓDULO 3: KALMAN ─────────────────────────────────────
    if mod_kalman:
        with st.expander("📡  MÓDULO 3 — Filtro de Kalman  (Precio Real sin Ruido)", expanded=True):
            st.markdown("""
            <div class="quantum-card">
            <div class="quantum-title">Filtro de Kalman — Ecuaciones de Riccati</div>
            Estima el <b>precio real subyacente</b> eliminando el ruido de mercado.
            <br>Modelo de estado: <b>x_t = [precio_real, velocidad]</b>
            <br>La <b>ganancia de Kalman K</b> balancea observación vs predicción de forma óptima.
            <br>Cuando precio cruza la línea Kalman = señal de tendencia real.
            </div>""", unsafe_allow_html=True)

            with st.spinner("Ejecutando filtro de Kalman..."):
                kal = kalman_filter(prices_arr)

            # Gráfico ancho M3 (sin columns)
            diff_k = prices_arr - kal["precio_kalman"]
            cruce_up   = np.where((diff_k[1:]>0)  & (diff_k[:-1]<=0))[0]+1
            cruce_down = np.where((diff_k[1:]<0)  & (diff_k[:-1]>=0))[0]+1
            fig_m3, axes_m3 = plt.subplots(2, 1, figsize=(13, 7), sharex=True, facecolor="#050810")
            fig_m3.subplots_adjust(hspace=0.1)
            n_k = len(prices_arr)
            axes_m3[0].plot(prices_arr, color="#1a3060", lw=1, alpha=0.7, label="Precio obs.")
            axes_m3[0].plot(kal["precio_kalman"], color="#00aaff", lw=2, label="Kalman (precio real)")
            axes_m3[0].fill_between(range(n_k), kal["banda_sup"], kal["banda_inf"], alpha=0.15, color="#0044ff", label="Banda 2σ")
            axes_m3[0].plot(kal["banda_sup"], color="#224488", lw=0.8, ls="--")
            axes_m3[0].plot(kal["banda_inf"], color="#224488", lw=0.8, ls="--")
            if len(cruce_up):
                axes_m3[0].scatter(cruce_up,   prices_arr[cruce_up],   color="#00ff88", s=50, zorder=5, marker="^", label="Cruce ↑ compra")
            if len(cruce_down):
                axes_m3[0].scatter(cruce_down, prices_arr[cruce_down], color="#ff3355", s=50, zorder=5, marker="v", label="Cruce ↓ venta")
            axes_m3[0].legend(fontsize=7, framealpha=0.3, loc="upper left")
            axes_m3[0].set_title("Precio Filtrado por Kalman + Señales de Cruce", color="#4488ff", fontsize=9)
            vel_k = kal["velocidad"]
            cv_k = ["#00ff88" if v>0 else "#ff3355" for v in vel_k]
            axes_m3[1].bar(range(n_k), vel_k, color=cv_k, alpha=0.7, width=0.8)
            axes_m3[1].axhline(0, color="#2a4060", lw=1, ls=":")
            axes_m3[1].set_title("Velocidad del Precio d[precio]/dt (Kalman)", color="#4488ff", fontsize=9)
            for ax_k in axes_m3:
                ax_k.set_facecolor("#050810"); ax_k.tick_params(colors="#2a4060")
                [sp.set_color("#0d1a2e") for sp in ax_k.spines.values()]
            st.pyplot(fig_m3); plt.close(fig_m3)
            # Métricas en fila
            t_col = {"ALCISTA ↑":"#00ff88","BAJISTA ↓":"#ff3355","LATERAL →":"#ffaa00"}.get(kal["tendencia"],"#aaa")
            mk1,mk2,mk3,mk4 = st.columns(4)
            mk1.markdown(f'<div class="metric-card" style="border-left:3px solid {t_col}"><div class="metric-label">Tendencia Kalman</div><div class="metric-value" style="color:{t_col}">{kal["tendencia"]}</div></div>', unsafe_allow_html=True)
            diff_c=kal["diff_pct"]; dc_col="#00ff88" if diff_c>0 else "#ff3355"
            mk2.markdown(f'<div class="metric-card"><div class="metric-label">Precio vs Kalman</div><div class="metric-value" style="color:{dc_col}">{diff_c:+.2f}%</div><div style="color:#2a4060;font-size:0.7rem">{"Sobre línea → alcista" if diff_c>0 else "Bajo línea → bajista"}</div></div>', unsafe_allow_html=True)
            mk3.markdown(f'<div class="metric-card"><div class="metric-label">Velocidad actual</div><div class="metric-value" style="color:{t_col}">{kal["vel_actual"]:+.6f}</div></div>', unsafe_allow_html=True)
            mk4.markdown(f'<div class="metric-card"><div class="metric-label">Cruces</div><div style="color:#00ff88;font-size:0.85rem">▲ Alcistas: {len(cruce_up)}</div><div style="color:#ff3355;font-size:0.85rem">▼ Bajistas: {len(cruce_down)}</div></div>', unsafe_allow_html=True)
    # ── MÓDULO 4: ENTRELAZAMIENTO ────────────────────────────
    if mod_entang:
        with st.expander("🔗  MÓDULO 4 — Entrelazamiento Cuántico de Activos", expanded=True):
            st.markdown("""
            <div class="quantum-card">
            <div class="quantum-title">Entropía de Von Neumann — S = −Tr(ρ ln ρ)</div>
            Calcula la <b>correlación cuántica</b> entre este activo y los mercados globales.
            <br><b>S≈0</b> → activo altamente entrelazado (se mueve con el mercado).
            <b>S grande</b> → activo independiente.
            <br>Un <b>espejo cuántico</b> (ρ≈−1) puede usarse como <i>cobertura natural</i>.
            </div>""", unsafe_allow_html=True)

            with st.spinner("Calculando entrelazamiento con mercados globales..."):
                ent = quantum_entanglement(df_raw, ticker_final)

            if ent is None:
                st.info("No se pudieron obtener suficientes activos de referencia para calcular el entrelazamiento.")
            else:
                # Gráfico ancho M4 (sin columns)
                fig_m4, axes_m4 = plt.subplots(1, 2, figsize=(13, 5), facecolor="#050810")
                nombres_e = ent["nombres"]; C_e = ent["C_matrix"]
                ax_e1 = axes_m4[0]; ax_e1.set_facecolor("#050810")
                cmap_e = LinearSegmentedColormap.from_list("q", ["#ff3355","#050810","#00ff88"])
                im_e = ax_e1.imshow(C_e, cmap=cmap_e, vmin=0, vmax=1, aspect="auto")
                ax_e1.set_xticks(range(len(nombres_e))); ax_e1.set_yticks(range(len(nombres_e)))
                ax_e1.set_xticklabels(nombres_e, rotation=45, ha="right", fontsize=7, color="#4a6080")
                ax_e1.set_yticklabels(nombres_e, fontsize=7, color="#4a6080")
                plt.colorbar(im_e, ax=ax_e1, shrink=0.8)
                ax_e1.set_title("Matriz de Densidad ρ — Entrelazamiento", color="#4488ff", fontsize=9)
                [sp.set_color("#0d1a2e") for sp in ax_e1.spines.values()]
                ax_e2 = axes_m4[1]; ax_e2.set_facecolor("#050810")
                corrs_e = [ent["correlaciones"][nm]["corr"] for nm in nombres_e]
                bar_ce  = ["#00ff88" if c>0.4 else("#ff3355" if c<-0.4 else "#4488ff") for c in corrs_e]
                bars_e  = ax_e2.barh(nombres_e, corrs_e, color=bar_ce, alpha=0.8, height=0.6)
                ax_e2.axvline(0,    color="#2a4060", lw=1)
                ax_e2.axvline(0.7,  color="#00ff88", lw=0.7, ls="--", alpha=0.5)
                ax_e2.axvline(-0.7, color="#ff3355", lw=0.7, ls="--", alpha=0.5)
                ax_e2.set_xlim(-1, 1)
                ax_e2.set_title(f"Correlación con {ticker_nombre}", color="#4488ff", fontsize=9)
                ax_e2.tick_params(colors="#2a4060", labelsize=8)
                [sp.set_color("#0d1a2e") for sp in ax_e2.spines.values()]
                for bar_e, co_e in zip(bars_e, corrs_e):
                    ax_e2.text(co_e+(0.03 if co_e>=0 else -0.03), bar_e.get_y()+bar_e.get_height()/2,
                               f"{co_e:.2f}", va="center", ha="left" if co_e>=0 else "right", color="white", fontsize=8)
                plt.tight_layout()
                st.pyplot(fig_m4); plt.close(fig_m4)
                # Métricas Von Neumann + tabla de correlaciones
                svn_pct = ent["S_norm"]*100
                svn_col = "#00ff88" if svn_pct<30 else ("#ffaa00" if svn_pct<60 else "#ff3355")
                me1, me2 = st.columns([1,2])
                with me1:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Entropía Von Neumann</div><div class="metric-value" style="color:{svn_col}">{ent["S_vn"]:.3f}</div><div class="score-bar-wrap"><div class="score-bar" style="width:{svn_pct:.0f}%;background:{svn_col}"></div></div><div style="color:#2a4060;font-size:0.7rem">{"Alta independencia" if svn_pct>60 else ("Entrelazado" if svn_pct<30 else "Correlación moderada")}</div></div>', unsafe_allow_html=True)
                with me2:
                    for nm_e, dat_e in ent["correlaciones"].items():
                        tipo_e, col_e = dat_e["tipo"]
                        co_e = dat_e["corr"]
                        st.markdown(f'<div class="metric-card" style="padding:6px 12px;border-left:3px solid {col_e}"><div style="display:flex;justify-content:space-between;font-size:0.78rem"><span style="color:#c0d0e0">{nm_e}</span><span style="color:{col_e}">{tipo_e}  ρ={co_e:.2f}</span></div></div>', unsafe_allow_html=True)
    # ── TABLA INDICADORES ────────────────────────────────────
    with st.expander("📊 Tabla de indicadores completa"):
        def fv(v,d=4): return f"{v:.{d}f}" if not pd.isna(v) else "N/D"
        tabla = {
            "Indicador":["RSI(14)","MACD","MACD Hist","BB %B","ATR %","EMA9","EMA21","EMA50","StochRSI K","Vol Ratio","Momentum 5p"],
            "Valor":[
                fv(ind["rsi"].iloc[-1],1), fv(ind["macd"].iloc[-1],6), fv(ind["macd_hist"].iloc[-1],6),
                fv(ind["bb_pct"].iloc[-1],3), f"{fv(ind['atr_pct'].iloc[-1],2)}%",
                fv(ind["ema9"].iloc[-1],4), fv(ind["ema21"].iloc[-1],4), fv(ind["ema50"].iloc[-1],4),
                fv(ind["stoch_k"].iloc[-1],1), f"{fv(ind['vol_ratio'].iloc[-1],2)}x",
                f"{fv(ind['momentum'].iloc[-1],2)}%"
            ]
        }
        st.dataframe(pd.DataFrame(tabla), use_container_width=True, hide_index=True)
