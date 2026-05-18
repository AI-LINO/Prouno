╔══════════════════════════════════════════════════════════════════╗
║           AI.LINO QUANTUM ENGINE v4 — GUÍA COMPLETA            ║
║         Para continuar desarrollo en chat nuevo de Claude       ║
╚══════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 CÓMO USAR ESTE ARCHIVO EN CHAT NUEVO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Abre chat nuevo con Claude
2. Primer mensaje: "Aquí está mi app.py completo de AI.Lino Quantum
   Engine v4. Lee el README.txt para contexto y continúa ayudándome."
3. Adjunta AMBOS archivos: app.py + este README.txt
4. Claude lee ambos sin deformar nada.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 QUÉ ES LA APP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dashboard de trading en Streamlit. Analiza acciones y criptos con:
- Indicadores técnicos clásicos (RSI, MACD, Bollinger, ATR, EMAs)
- Modelos matemáticos avanzados (HMM, Kalman, oscilador cuántico)
- Scanner de mercado para Bitso

Desplegada en: Streamlit Cloud (archivo app.py en GitHub)
Operador: trabaja en Bitso (cripto) y Yahoo Finance (acciones)
Mercados: Binance cripto, Yahoo acciones, CoinGecko (tokens nuevos)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ESTRUCTURA DEL CÓDIGO (app.py — 1762 líneas)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BLOQUE 1 — CONFIGURACIÓN (líneas 1-59)
  - st.set_page_config, CSS dark theme
  - Fuente: Orbitron + Rajdhani + Share Tech Mono
  - Clases CSS: .qcard .qtitle .sbox .mc .mv .ml .bw .bf

BLOQUE 2 — TIMEFRAMES (líneas 60-82)
  - TIMEFRAMES: dict con 6 opciones (1H·3d, 4H·10d, 1D·1m, etc.)
  - MTF_SUPERIOR: mapa para confirmación multi-timeframe

BLOQUE 3 — FUENTES DE DATOS (líneas 83-172)
  - binance_get_all_symbols() → todos los pares USDT/BTC/ETH/BNB
  - binance_buscar(q) → filtra pares por query
  - binance_descargar(symbol, interval, dias) → DataFrame OHLCV
  - coingecko_buscar(query) → busca coins por nombre
  - coingecko_descargar(coin_id, dias) → OHLCV desde CoinGecko
  - yahoo_descargar(ticker, interval, dias) → yfinance
  - cargar_datos(ticker, fuente, tf_key) → router principal
  - cargar_datos_superior(ticker, fuente, tf_key) → para MTF

BLOQUE 4 — INDICADORES TÉCNICOS (líneas 173-226)
  - calcular_rsi(c, p=14) → RSI con EWM (más reactivo)
  - calcular_macd(c) → MACD, signal, histograma
  - calcular_bb(c) → Bollinger Bands + %B
  - calcular_atr(h,l,c) → ATR con EWM
  - calcular_stoch_rsi(c) → Stochastic RSI
  - calcular_indicadores(df) → todos juntos + EMA9/21/50/200
    + VWAP + vol_ratio + momentum + ret_log + vol_ann + chg1 + chg3

BLOQUE 5 — FILTRO DE TENDENCIA (líneas 227-274)
  - filtro_tendencia(df, ind) → clasifica en 6 niveles:
    ALCISTA FUERTE / ALCISTA / ALCISTA DÉBIL /
    BAJISTA DÉBIL / BAJISTA / BAJISTA FUERTE / LATERAL
  - Usa EMA200 + alineación EMAs + pendiente EMA50
  - Retorna: tendencia, fuerza(0-100), bloquear_long, bloquear_short

BLOQUE 6 — MULTI-TIMEFRAME (líneas 275-307)
  - analisis_mtf(df_sup) → analiza timeframe superior
  - Retorna: tendencia_sup, score_sup(0-100), rsi_sup

BLOQUE 7 — HMM MEJORADO (líneas 308-350)
  - entrenar_hmm(df, ind) → 5 features: retorno, volatilidad,
    momentum, RSI normalizado, Bollinger %B
  - Normalización robusta: mediana + IQR (estable en cripto)
  - 4 semillas diferentes, penaliza estados vacíos (<7%)
  - n_iter=3000, covariance_type="full"
  - Estados: PÁNICO/BAJISTA(rojo), LATERAL(amarillo),
    ACUMULACIÓN(azul), ALCISTA/EUFORIA(verde)
  - urgencia: 1=alcista, 2=acumulación, 3=lateral, 4=pánico

BLOQUE 8 — SCORE COMPUESTO v4 (líneas 351-444)
  - calcular_score(ind, hmm_urgencia, tendencia_info, mtf_info)
  - 6 componentes: RSI(20pts) + MACD(18pts) + BB(12pts) +
    EMAs(18pts) + Volumen(12pts) + StochRSI(10pts)
  - Ajustes: HMM(+8/-18) + Tendencia(+10/-15) + MTF(+8/-10)
  - Bonus confluencia máxima: +15 si BB<0.15+RSI<35+vol>1.5
  - Retorna: score(0-100), desglose, ajustes_detalle

BLOQUE 9 — SEÑAL PRECISA + TRAILING (líneas 445-506)
  - generar_señal_precisa(score, ind, tend, mtf, fuente)
  - 5 tipos: LONG AGRESIVO(≥80) / LONG CONSERVADOR(≥68) /
    NEUTRAL(≥52) / PRECAUCIÓN(≥35) / SALIDA URGENTE(<35)
  - SL/TP calculados con ATR × multiplicador (cripto vs acciones)
  - Trailing dinámico: solo sube, nunca baja
  - Bloquea señales contra tendencia fuerte

BLOQUE 10 — BACKTESTING (líneas 507-594)
  - backtesting_simple(df, ind) → vectorizado sin lookahead bias
  - Estrategia: MACD cruce alcista + RSI<65 + EMA9>EMA21
  - Salida: trailing stop 1.5×ATR o cruce MACD bajista
  - Métricas: equity curve, win rate, profit factor, max drawdown

BLOQUE 11 — MÓDULOS CUÁNTICOS (líneas 595-699)
  - quantum_harmonic_oscillator(prices) → Ec. Schrödinger
    Niveles E_n = ℏω(n+½) → soportes/resistencias naturales
    |ψ|² → densidad de probabilidad del precio futuro
  - heisenberg_uncertainty(prices) → ΔxΔp ≥ ℏ/2
    U = Δx·Δp → confianza operativa, detección de tunnel cuántico
  - kalman_filter(prices) → ecuaciones de Riccati
    Estado [precio_real, velocidad], señales de cruce
  - quantum_entanglement(ticker, fuente) → entropía Von Neumann
    Correlación con BTC, ETH, S&P500, Oro, VIX, USD

BLOQUE 12 — UTILIDADES RENDER (líneas 700-711)
  - estilizar_ax(ax) → dark theme para matplotlib
  - render_fig(fig) → st.pyplot + plt.close (evita NULLs)
  ⚠️ IMPORTANTE: TODOS los gráficos deben usar render_fig()
     y estar FUERA de st.columns() para evitar bug NULL

BLOQUE 13 — SIDEBAR + UI PRINCIPAL (líneas 712-1176)
  - Buscador unificado: Yahoo + Binance + CoinGecko simultáneo
  - Favoritos: MON, BTC, ETH, SOL, PEPE, DOGE, NVDA, AAPL, TSLA, SPY
  - Timeframe selector + toggles módulos cuánticos + backtesting
  - Botón: ⚛️ ANALIZAR v4
  - Ejecución: descarga → indicadores → filtro tend → MTF → HMM
    → score → señal → gráfico 4 paneles → panel señal/SL/TP
    → backtesting → 4 módulos cuánticos colapsables → tabla

BLOQUE 14 — SCANNER BITSO (líneas 1179-1762)
  Fuente: CoinGecko (Binance bloqueado desde Streamlit Cloud)
  - COINS_BITSO: 30 pares disponibles en Bitso con IDs CoinGecko
  - cg_get_markets() → precios, cambios 1H/24H/7D, volumen
    @st.cache_data(ttl=120) — refresca cada 2 minutos
  - cg_get_ohlc(coin_id, days) → velas para RSI/BB/ATR/cascada
    @st.cache_data(ttl=300) — refresca cada 5 minutos
  - analizar_coin_cg(coin_data, ohlc_df) → análisis completo:
    · Cascada: velas verdes consecutivas
    · Mecha restante: RSI + BB%B + cambio 24H
    · Estabilidad: ATR% + consistencia retornos
    · Score 0-100: cascada+mecha+momentum+volumen+EMAs
  - UI con 4 tabs:
    🔥 Subiendo → cards con 🟩🟩🟩 velas + métricas
    📏 Más Estables → scatter ATR% vs Score + tabla
    🕯 Mecha Restante → barras horizontales de cuánto queda
    📋 Todos → tabla completa con progress bars
  - Selector final → abre par en motor principal

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 BUGS CONOCIDOS Y SOLUCIONES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BUG 1 — Gráficos NULL [0:NULL 1:NULL...]
  CAUSA: st.pyplot() dentro de st.columns() dentro de st.expander()
  SOLUCIÓN: Todos los gráficos van FUERA de st.columns().
  Las métricas van en columnas DEBAJO del gráfico.
  Siempre usar render_fig(fig) que hace plt.close() automático.

BUG 2 — %%writefile app.py en línea 1
  CAUSA: Es sintaxis de Jupyter Notebook, no Python puro.
  SOLUCIÓN: El archivo app.py NO debe tener esa línea.
  Si aparece, eliminar la primera línea antes de subir a GitHub.

BUG 3 — Scanner dice "No se pudieron obtener datos"
  CAUSA: Binance bloquea peticiones masivas desde Streamlit Cloud.
  SOLUCIÓN: El scanner usa CoinGecko. Si falla CoinGecko,
  esperar 60 segundos (rate limit free: ~30 req/min).

BUG 4 — CoinGecko rate limit 429
  CAUSA: Demasiadas peticiones seguidas.
  SOLUCIÓN: time.sleep(0.5) entre coins, @st.cache_data para
  no repetir llamadas. Si falla, esperar 1 minuto y reintentar.

BUG 5 — HMM no converge en criptos nuevas/muy volátiles
  CAUSA: Pocos datos o spikes extremos.
  SOLUCIÓN: Fallback silencioso con hmm_urg=2 (neutral).
  El análisis técnico sigue funcionando sin HMM.

BUG 6 — Módulo 4 Entrelazamiento: "sin suficientes activos"
  CAUSA: Con timeframes cortos (1H) los índices de tiempo no
  coinciden entre activos.
  SOLUCIÓN: El módulo siempre usa datos diarios de 90 días
  independiente del timeframe elegido.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DEPENDENCIAS (requirements.txt para Streamlit Cloud)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
streamlit
numpy
pandas
yfinance
matplotlib
hmmlearn
requests
scipy

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 MEJORAS PENDIENTES (para pedir en chat nuevo)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Alertas automáticas cuando un par llega a señal de compra
- Guardar historial de señales
- Comparar dos activos simultáneamente
- Exportar reporte PDF con el análisis
- Notificaciones por Telegram cuando el scanner encuentra oportunidades
- Agregar pares de BMV (mercado mexicano) para acciones MX

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 REGLAS IMPORTANTES PARA CLAUDE EN CHAT NUEVO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. NUNCA agregar %%writefile al inicio del archivo
2. SIEMPRE verificar sintaxis con ast.parse antes de entregar
3. SIEMPRE usar render_fig(fig) para gráficos, NUNCA st.pyplot directo
4. NUNCA poner st.pyplot() dentro de st.columns()
5. Para modificaciones: editar quirúrgicamente, no reescribir todo
6. El archivo completo son ~1762 líneas — entregarlo completo siempre
7. Fuente del scanner: CoinGecko, NO Binance (está bloqueado)
8. CSS dark theme: fondo #050810, acentos #4488ff/#00ff88/#ff3355

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 CONTEXTO DEL OPERADOR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Opera principalmente en Bitso (México)
- Mercados: cripto + acciones US + BMV
- Objetivo: encontrar subidas sostenidas de días/semanas
- No solo subidas rápidas de horas que luego caen
- Prefiere código completo listo para pegar, no fragmentos
- Despliega en Streamlit Cloud (no en local)
- Comunica errores con descripción de texto cuando no puede
  enviar imágenes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 BLOQUE 15 — AI.LINO LIVE (líneas 1763-2280)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Monitor en tiempo real de 20 pares.
Fuente: CoinGecko /coins/markets + /coins/{id}/ohlc (7 días, 4H)
Cache: markets=60s, ohlc=90s

Funciones:
- calcular_adx(h,l,c) → ADX + DI+ + DI-
- calcular_pendiente_ema(ema, ventana) → pendiente % últimas N velas
- analizar_live(coin_id, sym, mkt) → dict completo por par
- card_live(r) → HTML de card con barra fuerza + EMAs + ADX + RSI
- render_live() → ejecuta todo y renderiza 3 secciones

3 secciones:
✅ CASCADA FUERTE   → fuerza≥45 + acelerando + DI+>DI-
⚠️ PERDIENDO FUERZA → tiene cascada pero desacelerando
🔴 TENDENCIA BAJISTA→ DI->DI+ o pendientes negativas

Cada card muestra:
- Barra visual fuerza 0-100%
- EMA9/21/50 pendiente con flechas ↑↓ en color
- ADX, +DI, -DI, RSI
- ATR% del día
- Cambio 1H y 24H
- Velas verdes consecutivas 🟩🟩🟩

Auto-refresh: 30seg / 60seg / 2min / 5min / Manual
Botón sidebar: 📡 INICIAR MONITOR LIVE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 BLOQUE 16 — AI.LINO APEX (líneas 2351-3119)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sistema de entrada de alta precisión con 3 filtros en cascada.
Botón sidebar: 🚀 EJECUTAR APEX

FILTRO 1 — TENDENCIA BASE (0-100)
  EMA200 posición + alineación EMAs + ADX + DI+/DI- + pendiente EMA50
  Pasa si ≥ 55 puntos

FILTRO 2 — IMPULSO ACTIVO (0-100)
  MACD cruce/aceleración + Kalman vel+acc + Volumen spike + OBV + Cascada
  Pasa si ≥ 55 puntos

FILTRO 3 — ENTRADA PRECISA (0-100)
  RSI zona 28-58 + BB posición baja + Ruptura EMA21 + StochRSI
  Pasa si ≥ 55 puntos

Score APEX = F1×30% + F2×35% + F3×35%
  Bonus +12 si todos ≥ 70
  Cap en 45/50/48 si algún filtro falla

Clasificaciones:
  🚀 ENTRADA ÓPTIMA   → score ≥ 80, todos pasan
  ⚡ ENTRADA BUENA    → score ≥ 65, todos pasan
  👀 CASI LISTA       → F1+F2 pasan, F3 en progreso
  ⏳ EN FORMACIÓN     → solo F1 pasa
  ❌ NO CUMPLE        → ninguno pasa

Señales de salida automática:
  1. Mecha agotada: RSI>74 + BB>0.88 + ADX cayendo
  2. Tendencia rota: EMA9 cruza EMA21 hacia abajo
  3. Divergencia bajista: precio sube pero MACD+Kalman bajan
  4. MACD cruce bajista: histograma cruza a negativo

Niveles de entrada con ATR:
  Entry = precio actual
  SL = max(EMA21×0.992, precio - 1.5×ATR)
  TP1 = precio + 2.0×ATR  (objetivo conservador)
  TP2 = precio + 4.0×ATR  (objetivo moderado)
  TP3 = precio + 7.0×ATR  (dejar correr en tendencia fuerte)
  Trail = 1.5×ATR (trailing dinámico)

Botón "📡 Monitorear" → agrega al Live monitor + guarda en Supabase
Supabase tabla: ailino_apex_signals

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 BLOQUE 17 — SEGUIMIENTO PERSISTENTE + ALTO IMPACTO (líneas 3137+)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUPABASE tabla requerida:
CREATE TABLE IF NOT EXISTS ailino_posicion_activa (
  id SERIAL PRIMARY KEY, sym TEXT, cid TEXT,
  entry NUMERIC, sl NUMERIC, tp1 NUMERIC, tp2 NUMERIC, tp3 NUMERIC,
  trail NUMERIC, trail_sl NUMERIC, max_precio NUMERIC,
  atr_pct NUMERIC, score INTEGER, tiempo TEXT,
  activa BOOLEAN DEFAULT TRUE, estado TEXT DEFAULT 'SEGUIMIENTO',
  created_at TIMESTAMPTZ DEFAULT NOW()
);

SEGUIMIENTO PERSISTENTE:
- Se carga desde Supabase al abrir la app (persiste entre sesiones)
- Panel siempre visible arriba si hay posición activa
- Trail SL se actualiza automáticamente en Supabase
- Fuerza alcista 0-100% con semáforo: FUERTE/PERDIENDO/BAJISTA
- Señales de salida automáticas con urgencia 1-3
- Botón "✅ CERRÉ MI POSICIÓN" cierra en Supabase

FILTRO ALTO IMPACTO (filtro_alto_impacto):
- ATR% ≥ 2.5% (elimina centaveras)
- Score APEX ≥ 65
- R/R ≥ 2.0
- Vol ratio ≥ 1.3
- ADX ≥ 18
- Los 3 filtros pasan
- Ordena por: score × ATR% × R/R = "índice de impacto"

RANKING ALTO IMPACTO:
- Aparece automáticamente en APEX con las 3 mejores
- Medallas 🥇🥈🥉 con cards detalladas
- Botón "🎯 SEGUIR EN VIVO" guarda en Supabase directamente
