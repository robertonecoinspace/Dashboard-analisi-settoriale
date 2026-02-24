import numpy as np

# --- LOGICA DI ALLOCAZIONE AVANZATA ---
st.header("⚖️ Ripartizione Ottimizzata del Portafoglio")

# 1. Calcolo Volatilità (Rolling 6 mesi per stabilità)
returns_sectors = sector_prices.pct_change().dropna()
volatility = returns_sectors.std() * np.sqrt(252) # Annualizzata

# 2. Calcolo Inverse Volatility Weights
inv_vol = 1 / volatility
weights_risk = (inv_vol / inv_vol.sum()) * 100

# 3. Unione con Performance YTD (Filtro Momentum)
# Assegniamo peso solo ai settori con YTD > 0
positive_momentum = perf_ytd > 0
final_weights = weights_risk * positive_momentum
final_weights = (final_weights / final_weights.sum()) * 100

# Visualizzazione
col1, col2 = st.columns(2)
with col1:
    st.subheader("Pesi Finali (%)")
    st.bar_chart(final_weights)

with col2:
    st.subheader("Dettaglio Rischio/Rendimento")
    metrics_df = pd.DataFrame({
        'Volatilità (%)': (volatility * 100).round(2),
        'Perf YTD (%)': perf_ytd.round(2),
        'Peso (%)': final_weights.round(2)
    })
    st.dataframe(metrics_df)