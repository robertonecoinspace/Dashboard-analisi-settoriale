import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# 1. Configurazione Iniziale
st.set_page_config(page_title="Analisi Intermarket & Settoriale", layout="wide")

# 2. Definizioni Asset
intermarket_dict = {
    'Azionario USA (SPY)': 'SPY',
    'Oro (GLD)': 'GLD',
    'Obbligazionario 20Y+ (TLT)': 'TLT',
    'Dollaro Index (UUP)': 'UUP',
    'Bitcoin (BTC-USD)': 'BTC-USD'
}

sectors_dict = {
    'Tecnologia (XLK)': 'XLK',
    'Energia (XLE)': 'XLE',
    'Finanza (XLF)': 'XLF',
    'Salute (XLV)': 'XLV',
    'Consumi (XLY)': 'XLY',
    'Industria (XLI)': 'XLI',
    'Utility (XLU)': 'XLU',
    'Materiali (XLB)': 'XLB',
    'Real Estate (XLRE)': 'XLRE'
}

# 3. Funzione Download Dati (Robusta)
@st.cache_data(ttl=3600)
def fetch_financial_data(tickers):
    # auto_adjust=True sposta i dividendi nel prezzo 'Close' evitando l'errore 'Adj Close'
    data = yf.download(list(tickers), period="2y", auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        return data['Close']
    return data

# Unione ticker per scaricare tutto in una volta
all_tickers = list(intermarket_dict.values()) + list(sectors_dict.values())

# --- INIZIO APP ---
st.title("🏛️ Dashboard di Analisi Intermarket")
st.markdown("Analisi settimanale delle correlazioni e allocazione tattica di portafoglio.")

try:
    df_prices = fetch_financial_data(all_tickers)
    
    if df_prices.empty:
        st.error("Dati non disponibili. Controlla la connessione o i ticker.")
    else:
        # --- SEZIONE 1: CORRELAZIONI INTERMARKET ---
        st.header("🔄 Matrice di Correlazione Settimanale")
        inter_prices = df_prices[list(intermarket_dict.values())]
        # Resample a cadenza settimanale (W) e calcolo rendimenti
        weekly_returns = inter_prices.resample('W').last().pct_change().dropna()
        
        # Ridenominazione colonne per il grafico
        inv_inter = {v: k for k, v in intermarket_dict.items()}
        weekly_returns.columns = [inv_inter[c] for c in weekly_returns.columns]
        
        corr_matrix = weekly_returns.corr()
        fig_corr = px.imshow(
            corr_matrix, 
            text_auto=".2f", 
            color_continuous_scale='RdBu_r', 
            range_color=[-1, 1],
            title="Correlazione tra Asset Class"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # --- SEZIONE 2: PERFORMANCE SETTORIALE MULTI-FRAME ---
        st.divider()
        st.header("📈 Performance dei Settori")
        
        sec_prices = df_prices[list(sectors_dict.values())]
        
        # Calcolo timeframe (approssimato su giorni di borsa aperta)
        perf_df = pd.DataFrame()
        perf_df['Daily (%)'] = (sec_prices.iloc[-1] / sec_prices.iloc[-2] - 1) * 100
        perf_df['Weekly (%)'] = (sec_prices.iloc[-1] / sec_prices.iloc[-6] - 1) * 100
        perf_df['Monthly (%)'] = (sec_prices.iloc[-1] / sec_prices.iloc[-22] - 1) * 100
        
        # Calcolo YTD (Dall'inizio dell'anno corrente)
        start_year = f"{datetime.now().year}-01-01"
        first_price_year = sec_prices.loc[start_year:].iloc[0]
        perf_df['YTD (%)'] = (sec_prices.iloc[-1] / first_price_year - 1) * 100

        # Ridenominazione indici settori
        inv_sectors = {v: k for k, v in sectors_dict.items()}
        perf_df.index = [inv_sectors[i] for i in perf_df.index]
        
        st.dataframe(
            perf_df.style.format("{:.2f}%").background_gradient(cmap='RdYlGn', axis=0),
            use_container_width=True
        )

        # --- SEZIONE 3: PORTAFOGLIO RISK-PARITY (INVERSE VARIANCE) ---
        st.divider()
        st.header("⚖️ Ripartizione Ottimizzata del Portafoglio")
        st.info("Pesi calcolati in base all'inversa della volatilità (Risk Parity) filtrati per Momentum YTD positivo.")

        # Calcolo Volatilità annualizzata (Rolling 6 mesi per stabilità)
        vol = sec_prices.pct_change().std() * np.sqrt(252)
        inv_vol = 1 / vol
        
        # Filtro Momentum: Solo settori con YTD > 0 ricevono un peso
        ytd_values = (sec_prices.iloc[-1] / first_price_year - 1)
        weights = inv_vol * (ytd_values > 0)
        
        if weights.sum() > 0:
            final_weights = (weights / weights.sum()) * 100
            final_weights.index = [inv_sectors[i] for i in final_weights.index]
            
            c1, c2 = st.columns([2, 1])
            with c1:
                fig_pie = px.pie(
                    names=final_weights.index, 
                    values=final_weights.values, 
                    title="Allocazione Target %",
                    hole=0.4
                )
                st.plotly_chart(fig_pie)
            with c2:
                st.write("**Pesi di Portafoglio:**")
                st.table(final_weights.sort_values(ascending=False).rename("Peso %").round(2))
        else:
            st.warning("⚠️ Momentum Negativo: Nessun settore ha una performance YTD positiva. Allocazione non possibile.")

except Exception as e:
    st.error(f"Errore tecnico durante l'elaborazione: {e}")
    st.info("Suggerimento: Verifica che il file requirements.txt contenga 'yfinance', 'pandas', 'numpy', 'plotly' e 'streamlit'.")
