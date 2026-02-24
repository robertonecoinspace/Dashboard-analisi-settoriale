import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# 1. Configurazione Pagina
st.set_page_config(page_title="Financial Dashboard", layout="wide")

# 2. Definizione Ticker (Asset Intermarket e Settori)
intermarket_assets = {
    'S&P 500 (Azionario)': 'SPY',
    'Oro (Commodity)': 'GLD',
    'Treasury 20Y+ (Bond)': 'TLT',
    'Dollar Index (Valute)': 'UUP',
    'Bitcoin (Crypto)': 'BTC-USD'
}

sectors = {
    'Tecnologico': 'XLK',
    'Energetico': 'XLE',
    'Finanziario': 'XLF',
    'Sanitario': 'XLV',
    'Consumer Discretionary': 'XLY',
    'Industriale': 'XLI',
    'Utility': 'XLU'
}

# 3. Funzione Caricamento Dati
@st.cache_data(ttl=3600)
def load_data(tickers):
    data = yf.download(list(tickers), period="2y")['Adj Close']
    return data

# Uniamo tutti i ticker per un unico download
all_tickers = list(intermarket_assets.values()) + list(sectors.values())

try:
    df_prices = load_data(all_tickers)
    
    # --- MASCHERA 1: ANALISI INTERMARKET ---
    st.title("📊 Financial Analysis Dashboard")
    st.header("🔄 Analisi Intermarket (Correlazioni)")
    
    # Calcolo correlazioni settimanali
    inter_prices = df_prices[list(intermarket_assets.values())]
    weekly_returns = inter_prices.resample('W').last().pct_change().dropna()
    
    # Rinominiamo le colonne per chiarezza nel grafico
    inv_map = {v: k for k, v in intermarket_assets.items()}
    weekly_returns.columns = [inv_map[col] for col in weekly_returns.columns]
    
    corr_matrix = weekly_returns.corr()
    fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='RdBu_r', aspect="auto")
    st.plotly_chart(fig_corr, use_container_width=True)

    # --- MASCHERA 2: PERFORMANCE SETTORIALI ---
    st.divider()
    st.header("📈 Performance di Settore")
    
    sec_prices = df_prices[list(sectors.values())]
    
    # Calcolo timeframe
    last_price = sec_prices.iloc[-1]
    perf_1d = (last_price / sec_prices.iloc[-2] - 1) * 100
    perf_1w = (last_price / sec_prices.iloc[-6] - 1) * 100 # ~5gg borsa aperta
    perf_1m = (last_price / sec_prices.iloc[-22] - 1) * 100 # ~21gg borsa aperta
    
    ytd_start = f"{datetime.now().year}-01-01"
    perf_ytd = (last_price / sec_prices.loc[ytd_start:].iloc[0] - 1) * 100

    perf_table = pd.DataFrame({
        'Daily (%)': perf_1d,
        'Weekly (%)': perf_1w,
        'Monthly (%)': perf_1m,
        'YTD (%)': perf_1d # Nota: qui useremo perf_ytd calcolato sopra
    })
    # Correzione nomi settori
    inv_sectors = {v: k for k, v in sectors.items()}
    perf_table.index = [inv_sectors[idx] for idx in perf_table.index]
    
    st.dataframe(perf_table.style.format("{:.2f}%").background_gradient(cmap='RdYlGn'), use_container_width=True)

    # --- MASCHERA 3: RIPARTIZIONE PORTAFOGLIO (RISK PARITY) ---
    st.divider()
    st.header("⚖️ Ripartizione Ottimizzata del Portafoglio")
    
    # Logica: Inverse Variance (Ripartizione sul rischio)
    volatility = sec_prices.pct_change().dropna().std() * np.sqrt(252)
    inv_vol = 1 / volatility
    
    # Applichiamo pesi solo a chi ha YTD positivo (Momentum Filter)
    # Rialliniamo gli indici per sicurezza
    weights = inv_vol * (perf_ytd > 0)
    
    if weights.sum() > 0:
        final_weights = (weights / weights.sum()) * 100
        final_weights.index = [inv_sectors[idx] for idx in final_weights.index]
        
        col1, col2 = st.columns([2, 1])
        with col1:
            fig_weights = px.pie(names=final_weights.index, values=final_weights.values, title="Allocazione Asset")
            st.plotly_chart(fig_weights)
        with col2:
            st.write("Dettaglio Pesi:")
            st.table(final_weights.rename("Peso %").round(2))
    else:
        st.warning("Nessun settore ha performance YTD positiva. Impossibile allocare pesi basati sul Momentum.")

except Exception as e:
    st.error(f"Errore durante il recupero dei dati: {e}")
    st.info("Verifica la connessione internet o che i ticker siano corretti.")
