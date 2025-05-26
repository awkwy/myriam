import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# === Paramètres ===
ticker = "TSLA"  # Remplace par l’action de ton choix
today = datetime.today()
max_maturity = today + timedelta(days=730)  # 2 ans

# === Récupérer l'objet action ===
stock = yf.Ticker(ticker)

# === Récupérer toutes les dates d'expiration disponibles ===
expirations = stock.options

# === Filtrer les dates dans [aujourd’hui, aujourd’hui + 2 ans] ===
valid_expirations = [
    date for date in expirations
    if today <= datetime.strptime(date, "%Y-%m-%d") <= max_maturity
]

# === Récupérer toutes les options CALL valides ===
all_calls = []

for exp_date in valid_expirations:
    try:
        opt_chain = stock.option_chain(exp_date)
        calls = opt_chain.calls.copy()
        calls["expiration"] = exp_date
        all_calls.append(calls)
    except Exception as e:
        print(f"Erreur pour {exp_date} : {e}")

# === Concaténer les résultats ===
calls_df = pd.concat(all_calls, ignore_index=True)

# === Sauvegarder (optionnel) ===
calls_df.to_csv(f"{ticker}_calls_{today.date()}.csv", index=False)

# === Afficher un aperçu ===
print(calls_df[["contractSymbol", "strike", "expiration", "lastPrice", "impliedVolatility"]].head())
print(f"\nTotal de combinaisons CALL récupérées : {len(calls_df)}")


