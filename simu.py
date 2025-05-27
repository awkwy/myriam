import pandas as pd
import numpy as np
import math

# Paramètres de simulation de base
underlying_symbol = 'AAPL'       # Ticker du sous-jacent
initial_price = 150.0           # Prix initial du sous-jacent (en USD)
np.random.seed(42)              # Pour reproductibilité des aléas

# Générer 6 mois de dates de trading (environ 126 jours ouvrés)
start_date = pd.to_datetime('2024-01-02')
dates = pd.bdate_range(start_date, periods=126)  # 126 jours de bourse consécutifs

# Simuler les prix quotidiens du sous-jacent (processus aléatoire simple)
spot_prices = [initial_price]
for t in range(1, len(dates)):
    # Variation journalière aléatoire (rendement normal ~1% vol quotidien)
    daily_return = np.random.normal(0, 0.01)
    new_price = spot_prices[-1] * (1 + daily_return)
    spot_prices.append(new_price)
spot_prices = np.array(spot_prices)

# Définir quelques dates d'expiration fixes (≈1 mois, 3 mois, 6 mois, 1 an, 2 ans)
expirations = [
    start_date + pd.Timedelta(days=30),
    start_date + pd.Timedelta(days=90),
    start_date + pd.Timedelta(days=180),
    start_date + pd.Timedelta(days=365),
    start_date + pd.Timedelta(days=730)
]

# Définir une grille de strikes autour du prix initial (par ex. ±20% du spot initial)
strikes = np.arange(int(initial_price * 0.8), int(initial_price * 1.2) + 1, 5)
# (Ici de 120 à 180 par pas de 5 si initial_price=150)

# Simuler une volatilité de base ATM pour chaque jour (qui fluctue légèrement)
base_vols = [0.20]  # 20% vol implicite de base au départ (ATM)
for t in range(1, len(dates)):
    vol_prev = base_vols[-1]
    # Petit mouvement aléatoire de la vol de base, borné entre 10% et 50%
    vol_today = vol_prev + np.random.normal(0, 0.01)
    vol_today = min(0.50, max(0.10, vol_today))
    base_vols.append(vol_today)
base_vols = np.array(base_vols)

# Construire les données d'options
options_data = []
for i, current_date in enumerate(dates):
    S = spot_prices[i]               # Prix spot du jour
    base_vol = base_vols[i]          # Volatilité de base du jour (ATM, fraction)
    for exp_date in expirations:
        # Ignorer les expirations déjà passées ou ce jour même
        if exp_date <= current_date:
            continue
        days_to_exp = (exp_date - current_date).days
        # Filtrer les maturités hors [7, 730] jours
        if days_to_exp < 7 or days_to_exp > 730:
            continue
        T = days_to_exp / 365.0      # TTM en années (approximation 365j=1an)
        for K in strikes:
            # Calcul du sourire de volatilité
            moneyness = K / S
            smile_effect = 0.1 * (abs(math.log(moneyness)) ** 1.5)
            vol_random = np.random.normal(0, 0.02)
            iv_percent = (base_vol + smile_effect + vol_random) * 100  # en %
            # Limiter la vol implicite entre 5% et 200% pour éliminer valeurs extrêmes
            iv_percent = min(200.0, max(5.0, iv_percent))
            # Calcul du prix du CALL (valeur intrinsèque + valeur temps approximative)
            intrinsic_call = max(0.0, S - K)
            time_value = (iv_percent / 100.0) * S * math.sqrt(T) * 0.4
            call_price = max(0.01, intrinsic_call + time_value)  # prix min 0.01
            # Grecques approximatives pour le CALL
            delta_call = 0.5 + 0.4 * math.tanh((S - K) / (0.2 * S))
            gamma = 0.02 * math.exp(-0.5 * ((S - K) / (0.3 * S)) ** 2)
            theta = -0.1 * call_price / 365.0
            vega = 0.01 * S * math.sqrt(T)
            # Enregistrer le CALL
            options_data.append({
                'dataDate': current_date.strftime('%Y-%m-%d'),
                'optionType': 'CALL',
                'strike': float(K),
                'expirationDate': exp_date.strftime('%Y-%m-%d'),
                'lastPrice': call_price,
                'yearsToExpiration': T,
                'impliedVolatilityPct': iv_percent,
                'underlyingPrice': S,
                'delta': delta_call,
                'gamma': gamma,
                'theta': theta,
                'vega': vega
            })
            # Calcul du PUT correspondant (parité call-put)
            put_price = max(0.01, call_price + K - S)
            delta_put = delta_call - 1.0  # Delta put = delta call - 1
            # Enregistrer le PUT
            options_data.append({
                'dataDate': current_date.strftime('%Y-%m-%d'),
                'optionType': 'PUT',
                'strike': float(K),
                'expirationDate': exp_date.strftime('%Y-%m-%d'),
                'lastPrice': put_price,
                'yearsToExpiration': T,
                'impliedVolatilityPct': iv_percent,
                'underlyingPrice': S,
                'delta': delta_put,
                'gamma': gamma,
                'theta': theta,
                'vega': vega
            })

# Convertir en DataFrame pandas
df_options = pd.DataFrame(options_data)
# Exporter la DataFrame en CSV
output_file = f"{underlying_symbol}_simulated_options_6m_daily.csv"
df_options.to_csv(output_file, index=False)
print(f"✅ Données simulées générées: {len(df_options)} options sur {df_options['dataDate'].nunique()} jours. Fichier: {output_file}")

