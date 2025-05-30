"""
Script pédagogique de collecte et simulation de données d'options pour le pricing et la construction de surfaces de volatilité
Auteur: Assistant ChatGPT
Date: Mai 2025

Ce script simule la collecte des données d'options financières et leur structuration pour l'analyse pédagogique du pricing, 
des Greeks et des surfaces de volatilité.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, timedelta

# Paramètres pédagogiques initiaux
spot_price = 100  # Prix actuel de l'actif sous-jacent
risk_free_rate = 0.05  # Taux sans risque
volatility = 0.2  # Volatilité implicite initiale

# Définition des échéances et strikes
maturities_days = np.array([30, 90, 180, 365])  # maturités en jours
maturities_years = maturities_days / 365.25
strikes = np.array([90, 95, 100, 105, 110])

def black_scholes_price(S, K, T, r, sigma, option_type="CALL"):
    """
    Calcule le prix d'une option européenne avec le modèle de Black-Scholes.

    Inputs:
    - S (float): Prix spot du sous-jacent
    - K (float): Strike de l'option
    - T (float): Temps jusqu'à maturité (en années)
    - r (float): Taux sans risque
    - sigma (float): Volatilité implicite
    - option_type (str): 'CALL' ou 'PUT'

    Output:
    - price (float): Prix théorique de l'option

    Usage pratique:
    Cette formule est utile pour avoir une première approximation rapide du prix d'une option
    en supposant que les marchés sont efficients, sans frictions, et que la volatilité est constante.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "CALL":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def calculate_greeks(S, K, T, r, sigma, option_type="CALL"):
    """
    Calcule les sensibilités (Greeks) de l'option selon Black-Scholes.

    Inputs:
    - S (float): Prix spot
    - K (float): Strike
    - T (float): Temps à maturité (années)
    - r (float): Taux sans risque
    - sigma (float): Volatilité implicite
    - option_type (str): 'CALL' ou 'PUT'

    Output:
    - delta, gamma, theta, vega (floats): Sensibilités de l'option

    Usage pratique:
    Ces mesures permettent d'anticiper l'évolution du prix de l'option lorsque les conditions du marché
    changent. Utile pour les stratégies de couverture et de trading.

    Conseil:
    - Delta ≈ probabilité d'être in-the-money à maturité
    - Gamma montre la convexité du prix de l'option
    - Theta mesure la perte de valeur dans le temps
    - Vega donne l'impact d'un changement de volatilité
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = norm.cdf(d1) if option_type == "CALL" else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.25
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    return delta, gamma, theta, vega

"""
Étape 1 : Simulation des données d'options CALL
Ce bloc simule une grille complète de données d'options pour différentes échéances et strikes.
Les résultats sont stockés sous forme de dictionnaire structuré, enrichis avec les greeks.

Output: liste de dictionnaires (chaque dictionnaire = 1 option simulée)
"""
data_records = []
current_date = datetime.today()

for T_days, T in zip(maturities_days, maturities_years):
    expiration_date = current_date + timedelta(days=int(T_days))
    for K in strikes:
        option_price = black_scholes_price(spot_price, K, T, risk_free_rate, volatility)
        delta, gamma, theta, vega = calculate_greeks(spot_price, K, T, risk_free_rate, volatility)

        data_records.append({
            'dataDate': current_date.strftime('%Y-%m-%d'),
            'expirationDate': expiration_date.strftime('%Y-%m-%d'),
            'spotPrice': spot_price,
            'strike': K,
            'optionType': "CALL",
            'timeToMaturity': round(T, 4),
            'optionPrice': round(option_price, 2),
            'impliedVolatility': round(volatility * 100, 2),
            'delta': round(delta, 4),
            'gamma': round(gamma, 6),
            'theta': round(theta, 6),
            'vega': round(vega, 6)
        })

"""
Étape 2 : Structuration des résultats sous forme tabulaire avec pandas
Permet une visualisation claire des différentes options et leurs métriques associées
Output: DataFrame structuré pour analyse ou visualisation ultérieure
"""
df_options = pd.DataFrame(data_records)

# Affichage des données pédagogiques
print("\nDonnées simulées d'options pour la construction pédagogique de surfaces de volatilité :\n")
print(df_options.head(15))

# Sauvegarde éventuelle en CSV pour utilisation ultérieure
df_options.to_csv('simulated_options_data.csv', index=False)

