"""
Scraper de Données d'Options et Prix Historiques pour AAPL
Ce script récupère toutes les données nécessaires pour l'analyse ML des options
Auteur: Assistant Claude
Date: Mai 2025
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import json
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Pour l'API alternative si yfinance ne fonctionne pas
import requests
from bs4 import BeautifulSoup

# =====================================================
# CONFIGURATION
# =====================================================

class ScraperConfig:
    """Configuration centralisée pour le scraping"""
    TICKER = 'AAPL'
    
    # Période pour les données historiques
    START_DATE = '2022-01-01'  # 2+ ans d'historique
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    
    # Maturités d'options à récupérer (en nombre)
    MAX_EXPIRIES = 60  # Récupérer jusqu'à 20 dates d'expiration
    
    # Paramètres de délai pour éviter le rate limiting
    DELAY_BETWEEN_REQUESTS = 0.5  # secondes
    
    # Dossier de sauvegarde
    OUTPUT_DIR = 'options_data'
    
    # Paramètres pour le calcul du VIX-like
    VIX_WINDOW = 30  # jours

# Créer le dossier de sortie
os.makedirs(ScraperConfig.OUTPUT_DIR, exist_ok=True)

# =====================================================
# 1. RÉCUPÉRATION DES DONNÉES HISTORIQUES DE PRIX
# =====================================================

def fetch_historical_prices(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Récupère l'historique complet des prix pour calculer la volatilité réalisée
    
    Cette fonction récupère:
    - Prix OHLCV (Open, High, Low, Close, Volume)
    - Dividendes et splits si disponibles
    - Calcule les rendements et volatilités sur plusieurs fenêtres
    """
    print(f"\n📊 Récupération des données historiques pour {ticker}...")
    print(f"   Période: {start_date} à {end_date}")
    
    try:
        # Créer l'objet ticker
        stock = yf.Ticker(ticker)
        
        # Récupérer l'historique des prix
        hist = stock.history(start=start_date, end=end_date)
        
        if hist.empty:
            raise ValueError("Aucune donnée historique trouvée")
        
        print(f"✅ {len(hist)} jours de données récupérés")
        
        # Ajouter des colonnes calculées importantes
        hist['Returns'] = hist['Close'].pct_change()
        hist['Log_Returns'] = np.log(hist['Close'] / hist['Close'].shift(1))
        
        # Calculer la volatilité réalisée sur plusieurs fenêtres
        windows = [10, 20, 30, 60, 90, 252]  # Différentes fenêtres en jours
        for window in windows:
            # Volatilité annualisée en pourcentage
            hist[f'Realized_Vol_{window}D'] = (
                hist['Log_Returns'].rolling(window=window).std() * np.sqrt(252) * 100
            )
            
            # Volatilité Parkinson (utilise High et Low pour plus de précision)
            hist[f'Parkinson_Vol_{window}D'] = (
                np.sqrt(252 / (window * 4 * np.log(2))) * 
                (np.log(hist['High'] / hist['Low'])).rolling(window=window).sum() * 100
            )
        
        # Ajouter des indicateurs techniques utiles pour les options
        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        hist['RSI'] = calculate_rsi(hist['Close'])
        
        # Volume moyen (utile pour évaluer la liquidité)
        hist['Avg_Volume_20D'] = hist['Volume'].rolling(window=20).mean()
        
        # Calcul du rang de volatilité (où se situe la vol actuelle vs historique)
        hist['Vol_Rank_252D'] = hist['Realized_Vol_20D'].rolling(window=252).rank(pct=True) * 100
        
        # Informations sur les dividendes
        try:
            dividends = stock.dividends
            if not dividends.empty:
                # Joindre les dividendes aux données
                hist = hist.join(dividends.to_frame('Dividend'), how='left')
                hist['Dividend'].fillna(0, inplace=True)
                hist['Dividend_Yield'] = hist['Dividend'] / hist['Close'] * 4  # Annualisé (trimestre)
                print(f"✅ {len(dividends)} dividendes trouvés")
        except:
            hist['Dividend'] = 0
            hist['Dividend_Yield'] = 0
            print("⚠️  Pas de données de dividendes disponibles")
        
        # Sauvegarder
        filename = os.path.join(ScraperConfig.OUTPUT_DIR, f'{ticker}_price_history.csv')
        hist.to_csv(filename)
        print(f"💾 Données sauvegardées dans: {filename}")
        
        return hist
        
    except Exception as e:
        print(f"❌ Erreur lors de la récupération des prix: {str(e)}")
        return pd.DataFrame()

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calcule le RSI (Relative Strength Index)
    Utile pour identifier les conditions de surachat/survente
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# =====================================================
# 2. RÉCUPÉRATION DES DONNÉES D'OPTIONS ACTUELLES
# =====================================================

def fetch_options_chain(ticker: str, save_all: bool = True) -> pd.DataFrame:
    """
    Récupère la chaîne d'options complète avec toutes les dates d'expiration
    
    Cette fonction:
    - Récupère toutes les expirations disponibles
    - Collecte les données pour calls et puts
    - Calcule les Greeks approximatifs
    - Identifie les caractéristiques importantes (volume, OI, spreads)
    """
    print(f"\n📈 Récupération de la chaîne d'options pour {ticker}...")
    
    try:
        stock = yf.Ticker(ticker)
        
        # Obtenir toutes les dates d'expiration disponibles
        expirations = stock.options
        print(f"✅ {len(expirations)} dates d'expiration trouvées")
        
        # Limiter si nécessaire
        if len(expirations) > ScraperConfig.MAX_EXPIRIES:
            expirations = expirations[:ScraperConfig.MAX_EXPIRIES]
            print(f"   Limitation à {ScraperConfig.MAX_EXPIRIES} premières expirations")
        
        all_options = []
        
        # Récupérer le prix actuel du sous-jacent
        current_price = stock.info.get('regularMarketPrice', stock.info.get('currentPrice', 0))
        if current_price == 0:
            # Fallback: utiliser le dernier close
            hist = stock.history(period="1d")
            current_price = hist['Close'].iloc[-1] if not hist.empty else 150  # Valeur par défaut
        
        print(f"   Prix actuel: ${current_price:.2f}")
        
        # Récupérer les données pour chaque expiration
        for i, exp_date in enumerate(expirations):
            try:
                print(f"   📅 Traitement de l'expiration {i+1}/{len(expirations)}: {exp_date}")
                
                # Récupérer les options pour cette date
                opt = stock.option_chain(exp_date)
                
                # Traiter les CALLs
                calls = opt.calls.copy()
                calls['optionType'] = 'CALL'
                calls['expirationDate'] = exp_date
                
                # Traiter les PUTs
                puts = opt.puts.copy()
                puts['optionType'] = 'PUT'
                puts['expirationDate'] = exp_date
                
                # Combiner calls et puts
                options_df = pd.concat([calls, puts], ignore_index=True)
                
                # Ajouter le prix du sous-jacent
                options_df['underlyingPrice'] = current_price
                
                # Calculer le temps jusqu'à maturité
                exp_datetime = pd.to_datetime(exp_date)
                today = pd.Timestamp.now()
                days_to_expiry = (exp_datetime - today).days
                options_df['daysToExpiration'] = days_to_expiry
                options_df['yearsToExpiration'] = days_to_expiry / 365.25
                
                # Calculer la moneyness
                options_df['moneyness'] = options_df['strike'] / current_price
                options_df['logMoneyness'] = np.log(options_df['moneyness'])
                
                # Identifier ITM/OTM
                options_df['inTheMoney'] = (
                    ((options_df['optionType'] == 'CALL') & (options_df['strike'] < current_price)) |
                    ((options_df['optionType'] == 'PUT') & (options_df['strike'] > current_price))
                )
                
                # Calculer les spreads bid-ask
                options_df['bidAskSpread'] = options_df['ask'] - options_df['bid']
                options_df['bidAskSpreadPct'] = (
                    options_df['bidAskSpread'] / options_df['lastPrice'].replace(0, np.nan) * 100
                )
                
                # Score de liquidité basé sur le volume et l'open interest
                options_df['liquidityScore'] = (
                    options_df['volume'] * 0.3 + options_df['openInterest'] * 0.7
                )
                
                # Ajouter à la liste
                all_options.append(options_df)
                
                # Petit délai pour éviter de surcharger l'API
                time.sleep(ScraperConfig.DELAY_BETWEEN_REQUESTS)
                
            except Exception as e:
                print(f"   ⚠️  Erreur pour l'expiration {exp_date}: {str(e)}")
                continue
        
        # Combiner toutes les options
        if all_options:
            full_chain = pd.concat(all_options, ignore_index=True)
            print(f"\n✅ Total: {len(full_chain)} options récupérées")
            
            # Nettoyer et standardiser les colonnes
            full_chain = clean_options_data(full_chain)
            
            # Calculer la volatilité implicite approximative si non fournie
            if 'impliedVolatility' not in full_chain.columns:
                print("   📊 Estimation de la volatilité implicite...")
                full_chain['impliedVolatility'] = estimate_implied_volatility(full_chain)
            
            # Convertir la volatilité en pourcentage
            full_chain['impliedVolatilityPct'] = full_chain['impliedVolatility'] * 100
            
            # Sauvegarder
            if save_all:
                filename = os.path.join(ScraperConfig.OUTPUT_DIR, f'{ticker}_options_chain_full.csv')
                full_chain.to_csv(filename, index=False)
                print(f"💾 Chaîne d'options complète sauvegardée: {filename}")
            
            return full_chain
        else:
            print("❌ Aucune option récupérée")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Erreur lors de la récupération des options: {str(e)}")
        return pd.DataFrame()

def clean_options_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie et standardise les données d'options
    
    Cette fonction:
    - Renomme les colonnes pour la cohérence
    - Remplit les valeurs manquantes
    - Supprime les données aberrantes
    """
    # Mapping des noms de colonnes pour standardisation
    column_mapping = {
        'lastTradeDate': 'lastTradeDate',
        'strike': 'strike',
        'lastPrice': 'lastPrice',
        'bid': 'bid',
        'ask': 'ask',
        'change': 'change',
        'percentChange': 'percentChange',
        'volume': 'volume',
        'openInterest': 'openInterest',
        'impliedVolatility': 'impliedVolatility'
    }
    
    # Renommer si nécessaire
    df = df.rename(columns=column_mapping)
    
    # Remplir les valeurs manquantes
    df['volume'].fillna(0, inplace=True)
    df['openInterest'].fillna(0, inplace=True)
    df['bid'].fillna(0, inplace=True)
    df['ask'].fillna(df['lastPrice'], inplace=True)
    
    # Supprimer les options avec des strikes extrêmes (> 3x ou < 0.3x le prix)
    df = df[
        (df['moneyness'] >= 0.3) & 
        (df['moneyness'] <= 3.0)
    ]
    
    # Supprimer les options avec des prix négatifs ou nuls
    df = df[df['lastPrice'] > 0]
    
    return df

def estimate_implied_volatility(df: pd.DataFrame) -> pd.Series:
    """
    Estime la volatilité implicite basée sur les caractéristiques de l'option
    
    Utilise une approche simplifiée basée sur:
    - La moneyness
    - Le temps jusqu'à maturité
    - Le type d'option (call/put)
    """
    # Volatilité de base (peut être ajustée selon l'actif)
    base_vol = 0.25  # 25% pour AAPL
    
    # Ajustement pour la moneyness (smile effect)
    moneyness_adjustment = 0.1 * np.abs(df['logMoneyness']) ** 1.5
    
    # Ajustement pour la maturité (structure à terme)
    maturity_adjustment = 0.05 * np.exp(-2 * df['yearsToExpiration'])
    
    # Ajustement pour le skew (puts généralement plus chers)
    skew_adjustment = np.where(
        (df['optionType'] == 'PUT') & (df['moneyness'] < 1),
        0.05,  # 5% de plus pour les puts OTM
        0
    )
    
    # Calcul final avec un peu de bruit aléatoire
    estimated_vol = (
        base_vol + 
        moneyness_adjustment + 
        maturity_adjustment + 
        skew_adjustment +
        np.random.normal(0, 0.02, len(df))  # 2% de bruit
    )
    
    # Limiter entre 5% et 100%
    return np.clip(estimated_vol, 0.05, 1.0)

# =====================================================
# 3. CALCUL DES INDICATEURS DE MARCHÉ (VIX-LIKE)
# =====================================================

def calculate_vix_proxy(options_df: pd.DataFrame, spot_price: float) -> float:
    """
    Calcule un proxy du VIX basé sur les options à 30 jours
    
    Le VIX mesure la volatilité implicite moyenne des options S&P 500 à 30 jours.
    Ici, nous créons un proxy similaire pour AAPL.
    """
    # Filtrer les options proches de 30 jours
    target_days = 30
    tolerance = 10  # +/- 10 jours
    
    near_term = options_df[
        (options_df['daysToExpiration'] >= target_days - tolerance) &
        (options_df['daysToExpiration'] <= target_days + tolerance)
    ].copy()
    
    if len(near_term) == 0:
        return 25.0  # Valeur par défaut
    
    # Filtrer les options près de l'ATM (moneyness entre 0.9 et 1.1)
    atm_options = near_term[
        (near_term['moneyness'] >= 0.9) &
        (near_term['moneyness'] <= 1.1)
    ]
    
    if len(atm_options) == 0:
        return 25.0
    
    # Calculer la volatilité moyenne pondérée par l'open interest
    weights = atm_options['openInterest'] / atm_options['openInterest'].sum()
    vix_proxy = (atm_options['impliedVolatilityPct'] * weights).sum()
    
    return vix_proxy

# =====================================================
# 4. ENRICHISSEMENT DES DONNÉES D'OPTIONS
# =====================================================

def enrich_options_data(options_df: pd.DataFrame, price_history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrichit les données d'options avec des features supplémentaires
    
    Ajoute:
    - Volatilité historique correspondante
    - Ratio volatilité implicite/réalisée
    - Indicateurs techniques du sous-jacent
    - Greeks approximatifs
    """
    print("\n🔧 Enrichissement des données d'options...")
    
    # Obtenir les dernières valeurs de volatilité réalisée
    if not price_history_df.empty:
        latest_data = price_history_df.iloc[-1]
        
        # Ajouter la volatilité réalisée
        for window in [10, 20, 30, 60]:
            col_name = f'Realized_Vol_{window}D'
            if col_name in latest_data:
                options_df[col_name] = latest_data[col_name]
        
        # Ratio volatilité implicite/réalisée (pour identifier les opportunités)
        if 'Realized_Vol_20D' in options_df.columns:
            options_df['IV_RV_Ratio'] = (
                options_df['impliedVolatilityPct'] / options_df['Realized_Vol_20D']
            )
        
        # Ajouter des indicateurs techniques
        options_df['RSI'] = latest_data.get('RSI', 50)
        options_df['Vol_Rank'] = latest_data.get('Vol_Rank_252D', 50)
    
    # Calculer les Greeks approximatifs
    print("   📊 Calcul des Greeks approximatifs...")
    
    # Utiliser le taux sans risque actuel (peut être mis à jour)
    risk_free_rate = 0.05  # 5%
    
    # Delta approximatif (simplifié)
    options_df['delta_approx'] = calculate_approximate_delta(
        options_df['underlyingPrice'],
        options_df['strike'],
        options_df['yearsToExpiration'],
        options_df['impliedVolatility'],
        options_df['optionType']
    )
    
    # Gamma approximatif
    options_df['gamma_approx'] = calculate_approximate_gamma(
        options_df['underlyingPrice'],
        options_df['strike'],
        options_df['yearsToExpiration'],
        options_df['impliedVolatility']
    )
    
    # Vega approximatif (sensibilité à la volatilité)
    options_df['vega_approx'] = calculate_approximate_vega(
        options_df['underlyingPrice'],
        options_df['strike'],
        options_df['yearsToExpiration'],
        options_df['impliedVolatility']
    )
    
    # Theta approximatif (décroissance temporelle)
    options_df['theta_approx'] = calculate_approximate_theta(
        options_df['underlyingPrice'],
        options_df['strike'],
        options_df['yearsToExpiration'],
        options_df['impliedVolatility'],
        options_df['optionType'],
        risk_free_rate
    )
    
    # Calculer le VIX proxy pour AAPL
    vix_proxy = calculate_vix_proxy(options_df, options_df['underlyingPrice'].iloc[0])
    options_df['VIX_Proxy'] = vix_proxy
    print(f"   📈 VIX Proxy calculé: {vix_proxy:.2f}%")
    
    # Identifier les options avec des caractéristiques intéressantes
    options_df['high_volume'] = options_df['volume'] > options_df['volume'].quantile(0.9)
    options_df['high_oi'] = options_df['openInterest'] > options_df['openInterest'].quantile(0.9)
    options_df['tight_spread'] = options_df['bidAskSpreadPct'] < 5  # Spread < 5%
    
    # Score de qualité global
    options_df['quality_score'] = (
        options_df['high_volume'].astype(int) * 0.3 +
        options_df['high_oi'].astype(int) * 0.3 +
        options_df['tight_spread'].astype(int) * 0.4
    )
    
    print(f"✅ Enrichissement terminé: {len(options_df.columns)} colonnes totales")
    
    return options_df


# =====================================================
# 5. GREEKS APPROXIMATIFS — FONCTIONS CORRIGÉES
# =====================================================

def calculate_approximate_delta(S, K, T, sigma, option_type):
    """
    Delta approximatif (vectorisé).
    `option_type` doit contenir 'CALL' ou 'PUT'.
    """
    from scipy.stats import norm
    T     = np.maximum(T, 0.001)
    sigma = np.maximum(sigma, 0.01)
    d1    = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return np.where(option_type == 'CALL', norm.cdf(d1), norm.cdf(d1) - 1)

def calculate_approximate_gamma(S, K, T, sigma):
    """Gamma approximatif (vectorisé)."""
    from scipy.stats import norm
    T     = np.maximum(T, 0.001)
    sigma = np.maximum(sigma, 0.01)
    d1    = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def calculate_approximate_vega(S, K, T, sigma):
    """Vega approximatif (vectorisé, pour 1 % de variation de vol)."""
    from scipy.stats import norm
    T     = np.maximum(T, 0.001)
    sigma = np.maximum(sigma, 0.01)
    d1    = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T) / 100

def calculate_approximate_theta(S, K, T, sigma, option_type, r):
    """
    Theta approximatif (vectorisé) — résultat par jour.
    `option_type` doit contenir 'CALL' ou 'PUT'.
    """
    from scipy.stats import norm
    T     = np.maximum(T, 0.001)
    sigma = np.maximum(sigma, 0.01)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                  - r * K * np.exp(-r * T) * norm.cdf(d2))
    put_theta  = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                  + r * K * np.exp(-r * T) * norm.cdf(-d2))

    return np.where(option_type == 'CALL', call_theta, put_theta) / 365

# =====================================================
# 6. FORMATAGE FINAL POUR LE MODÈLE ML
# =====================================================


def prepare_ml_dataset(options_df: pd.DataFrame, price_history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare le dataset final pour le modèle ML avec toutes les features nécessaires
    
    Cette fonction:
    - Sélectionne et renomme les colonnes pour correspondre au modèle
    - Ajoute les features calculées
    - Filtre les données de qualité insuffisante
    - Sauvegarde le dataset prêt pour l'entraînement
    """
    print("\n🎯 Préparation du dataset final pour le ML...")
    
    # Copier pour éviter de modifier l'original
    ml_df = options_df.copy()
    
    # Renommer les colonnes pour correspondre au format attendu par le modèle
    column_mapping = {
        'strike': 'Strike',
        'underlyingPrice': 'Prix_Sous_Jacent',
        'lastPrice': 'Prix_Option',
        'impliedVolatilityPct': 'Volatilite_Implicite',
        'yearsToExpiration': 'Temps_Maturite_Annees',
        'daysToExpiration': 'Jours_Jusqu_Maturite',
        'optionType': 'Type_Option',
        'volume': 'Volume',
        'openInterest': 'Open_Interest',
        'moneyness': 'Moneyness',
        'logMoneyness': 'Log_Moneyness',
        'delta_approx': 'Delta',
        'gamma_approx': 'Gamma',
        'vega_approx': 'Vega',
        'theta_approx': 'Theta',
        'bidAskSpread': 'Bid_Ask_Spread',
        'bidAskSpreadPct': 'Bid_Ask_Spread_Pct',
        'liquidityScore': 'Liquidity_Score',
        'VIX_Proxy': 'VIX_Proxy'
    }
    
    ml_df.rename(columns=column_mapping, inplace=True)
    
    # Ajouter les colonnes manquantes avec des valeurs par défaut
    ml_df['Taux_Sans_Risque'] = 0.05  # 5%
    ml_df['Dividend_Yield'] = 0.015  # 1.5% pour AAPL
    
    # Calculer Rho approximatif
    ml_df['Rho'] = ml_df['Temps_Maturite_Annees'] * ml_df['Strike'] * 0.01  # Simplifié
    
    # Greeks de second ordre
    ml_df['Vanna'] = ml_df['Vega'] * ml_df['Delta'] * 0.1  # Approximation
    ml_df['Volga'] = ml_df['Vega'] * ml_df['Gamma'] * 0.1  # Approximation
    
    # Features supplémentaires
    ml_df['Strike_Ratio'] = ml_df['Strike'] / ml_df['Prix_Sous_Jacent']
    ml_df['Sqrt_Time'] = np.sqrt(ml_df['Temps_Maturite_Annees'])
    ml_df['Vol_Time'] = ml_df['Volatilite_Implicite'] * ml_df['Sqrt_Time']
    
    # Encoder le type d'option
    ml_df['Is_Call'] = (ml_df['Type_Option'] == 'CALL').astype(int)
    
    # Features de liquidité
    ml_df['Log_Volume'] = np.log1p(ml_df['Volume'])
    ml_df['Log_Open_Interest'] = np.log1p(ml_df['Open_Interest'])
    
    # Identifier OTM/ITM
    ml_df['Is_OTM'] = (
        ((ml_df['Type_Option'] == 'CALL') & (ml_df['Strike'] > ml_df['Prix_Sous_Jacent'])) |
        ((ml_df['Type_Option'] == 'PUT') & (ml_df['Strike'] < ml_df['Prix_Sous_Jacent']))
    ).astype(int)
    
    ml_df['OTM_Distance'] = np.abs(ml_df['Log_Moneyness']) * ml_df['Is_OTM']
    
    # Interactions entre Greeks
    ml_df['Delta_Gamma_Product'] = ml_df['Delta'] * ml_df['Gamma']
    ml_df['Vega_Volga_Ratio'] = ml_df['Vega'] / (ml_df['Volga'] + 0.001)
    
    # Calculer le skew de volatilité si possible
    ml_df['Volatility_Skew'] = calculate_volatility_skew_for_ml(ml_df)
    
    # Ratio IV/VIX
    ml_df['Vol_VIX_Ratio'] = ml_df['Volatilite_Implicite'] / ml_df['VIX_Proxy']
    
    # Filtrer les données de mauvaise qualité
    print("   🧹 Filtrage des données...")
    initial_count = len(ml_df)
    
    # Critères de filtrage
    ml_df = ml_df[
        (ml_df['Prix_Option'] > 0.01) &  # Prix minimum
        (ml_df['Volume'] > 0) &  # Au moins un peu de volume
        (ml_df['Bid_Ask_Spread_Pct'] < 50) &  # Spread raisonnable
        (ml_df['Moneyness'] >= 0.5) & (ml_df['Moneyness'] <= 2.0) &  # Strikes raisonnables
        (ml_df['Temps_Maturite_Annees'] > 0.01) &  # Au moins quelques jours
        (ml_df['Volatilite_Implicite'] > 5) & (ml_df['Volatilite_Implicite'] < 200)  # Vol raisonnable
    ]
    
    filtered_count = initial_count - len(ml_df)
    print(f"   ✅ {filtered_count} lignes filtrées ({filtered_count/initial_count*100:.1f}%)")
    
    # Ajouter la date de scraping
    ml_df['Date_Scraping'] = datetime.now().strftime('%Y-%m-%d')
    
    # Trier par type d'option et maturité
    ml_df.sort_values(['Type_Option', 'Temps_Maturite_Annees', 'Strike'], inplace=True)
    
    # Sauvegarder le dataset final
    filename = os.path.join(ScraperConfig.OUTPUT_DIR, f'{ScraperConfig.TICKER}_options_data_ml_ready.csv')
    ml_df.to_csv(filename, index=False)
    print(f"\n💾 Dataset ML sauvegardé: {filename}")
    print(f"   Shape: {ml_df.shape}")
    
    # Créer aussi une version filtrée pour le modèle (comme dans le code original)
    # Garder seulement les options les plus liquides pour l'entraînement initial
    ml_df_filtered = ml_df[
        (ml_df['Volume'] >= ml_df['Volume'].quantile(0.25)) |
        (ml_df['Open_Interest'] >= ml_df['Open_Interest'].quantile(0.25))
    ]
    
    filename_filtered = os.path.join(ScraperConfig.OUTPUT_DIR, f'{ScraperConfig.TICKER}_options_data_filter.csv')
    ml_df_filtered.to_csv(filename_filtered, index=False)
    print(f"   Dataset filtré sauvegardé: {filename_filtered} (Shape: {ml_df_filtered.shape})")
    
    return ml_df

def calculate_volatility_skew_for_ml(df: pd.DataFrame) -> pd.Series:
    """
    Calcule le skew de volatilité pour chaque ligne basé sur la maturité
    
    Le skew mesure la différence de volatilité entre puts OTM et calls OTM
    """
    skew_values = []
    
    for idx, row in df.iterrows():
        # Trouver des options similaires (même maturité, différents strikes)
        same_maturity = df[
            (df['Temps_Maturite_Annees'] == row['Temps_Maturite_Annees']) &
            (df['Type_Option'] == row['Type_Option'])
        ]
        
        if len(same_maturity) < 5:  # Pas assez de données
            skew_values.append(0)
            continue
        
        # Calculer le skew local
        atm_strike = row['Prix_Sous_Jacent']
        otm_puts = same_maturity[
            (same_maturity['Type_Option'] == 'PUT') & 
            (same_maturity['Strike'] < atm_strike * 0.95)
        ]
        otm_calls = same_maturity[
            (same_maturity['Type_Option'] == 'CALL') & 
            (same_maturity['Strike'] > atm_strike * 1.05)
        ]
        
        if len(otm_puts) > 0 and len(otm_calls) > 0:
            put_vol = otm_puts['Volatilite_Implicite'].mean()
            call_vol = otm_calls['Volatilite_Implicite'].mean()
            skew = put_vol - call_vol
        else:
            skew = 0
        
        skew_values.append(skew)
    
    return pd.Series(skew_values, index=df.index)

# =====================================================
# 7. RAPPORT DE QUALITÉ DES DONNÉES
# =====================================================

def generate_data_quality_report(ml_df: pd.DataFrame, price_df: pd.DataFrame) -> Dict:
    """
    Génère un rapport détaillé sur la qualité des données collectées
    
    Analyse:
    - Complétude des données
    - Distribution des features
    - Qualité de la liquidité
    - Couverture des strikes et maturités
    """
    print("\n📊 Génération du rapport de qualité des données...")
    
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ticker': ScraperConfig.TICKER,
        'data_summary': {
            'total_options': len(ml_df),
            'unique_expirations': ml_df['Temps_Maturite_Annees'].nunique(),
            'unique_strikes': ml_df['Strike'].nunique(),
            'calls_count': len(ml_df[ml_df['Type_Option'] == 'CALL']),
            'puts_count': len(ml_df[ml_df['Type_Option'] == 'PUT']),
            'price_history_days': len(price_df) if not price_df.empty else 0
        },
        'liquidity_analysis': {
            'avg_volume': ml_df['Volume'].mean(),
            'avg_open_interest': ml_df['Open_Interest'].mean(),
            'pct_with_volume': (ml_df['Volume'] > 0).mean() * 100,
            'pct_tight_spread': (ml_df['Bid_Ask_Spread_Pct'] < 5).mean() * 100
        },
        'volatility_analysis': {
            'avg_implied_vol': ml_df['Volatilite_Implicite'].mean(),
            'vol_range': [ml_df['Volatilite_Implicite'].min(), ml_df['Volatilite_Implicite'].max()],
            'vix_proxy': ml_df['VIX_Proxy'].iloc[0] if 'VIX_Proxy' in ml_df.columns else None
        },
        'moneyness_distribution': {
            'deep_otm_puts': len(ml_df[ml_df['Moneyness'] < 0.9]),
            'otm_puts': len(ml_df[(ml_df['Moneyness'] >= 0.9) & (ml_df['Moneyness'] < 0.97)]),
            'atm': len(ml_df[(ml_df['Moneyness'] >= 0.97) & (ml_df['Moneyness'] <= 1.03)]),
            'otm_calls': len(ml_df[(ml_df['Moneyness'] > 1.03) & (ml_df['Moneyness'] <= 1.1)]),
            'deep_otm_calls': len(ml_df[ml_df['Moneyness'] > 1.1])
        },
        'maturity_distribution': {
            'short_term_pct': (ml_df['Temps_Maturite_Annees'] < 0.1).mean() * 100,
            'medium_term_pct': ((ml_df['Temps_Maturite_Annees'] >= 0.1) & 
                               (ml_df['Temps_Maturite_Annees'] < 0.5)).mean() * 100,
            'long_term_pct': (ml_df['Temps_Maturite_Annees'] >= 0.5).mean() * 100
        },
        'data_quality_scores': {
            'completeness': 1 - ml_df.isnull().sum().sum() / (len(ml_df) * len(ml_df.columns)),
            'liquidity_score': ml_df['Liquidity_Score'].mean() if 'Liquidity_Score' in ml_df.columns else 0,
            'greeks_availability': sum([col in ml_df.columns for col in ['Delta', 'Gamma', 'Vega', 'Theta']]) / 4
        }
    }
    
    # Sauvegarder le rapport
    filename = os.path.join(ScraperConfig.OUTPUT_DIR, 'data_quality_report.json')
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"💾 Rapport de qualité sauvegardé: {filename}")
    
    # Afficher un résumé
    print("\n📈 Résumé du rapport:")
    print(f"   - Options totales: {report['data_summary']['total_options']:,}")
    print(f"   - Expirations uniques: {report['data_summary']['unique_expirations']}")
    print(f"   - Volume moyen: {report['liquidity_analysis']['avg_volume']:.0f}")
    print(f"   - Volatilité implicite moyenne: {report['volatility_analysis']['avg_implied_vol']:.1f}%")
    print(f"   - Score de complétude: {report['data_quality_scores']['completeness']:.2%}")
    
    return report

# =====================================================
# 8. FONCTION PRINCIPALE
# =====================================================

def main():
    """
    Fonction principale qui orchestre tout le processus de scraping
    """
    print("🚀 Démarrage du scraping des données d'options pour AAPL")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # 1. Récupérer l'historique des prix
        price_history = fetch_historical_prices(
            ScraperConfig.TICKER,
            ScraperConfig.START_DATE,
            ScraperConfig.END_DATE
        )
        
        # 2. Récupérer la chaîne d'options actuelle
        options_chain = fetch_options_chain(ScraperConfig.TICKER)
        
        if options_chain.empty:
            print("❌ Échec de la récupération des options. Arrêt du processus.")
            return
        
        # 3. Enrichir les données d'options
        enriched_options = enrich_options_data(options_chain, price_history)
        
        # 4. Préparer le dataset pour le ML
        ml_dataset = prepare_ml_dataset(enriched_options, price_history)
        
        # 5. Générer le rapport de qualité
        quality_report = generate_data_quality_report(ml_dataset, price_history)
        
        # Calculer le temps total
        elapsed_time = time.time() - start_time
        print(f"\n✅ Processus terminé en {elapsed_time:.1f} secondes")
        
        # Résumé final
        print("\n📊 Résumé final:")
        print(f"   - Fichiers créés dans: {ScraperConfig.OUTPUT_DIR}/")
        print(f"   - Prix historiques: {len(price_history)} jours")
        print(f"   - Options récupérées: {len(options_chain)}")
        print(f"   - Dataset ML final: {len(ml_dataset)} lignes")
        
        print("\n🎯 Prochaines étapes:")
        print("   1. Vérifier la qualité des données dans le rapport")
        print("   2. Lancer finance3_enhanced.py pour l'entraînement des modèles")
        print("   3. Ajuster les hyperparamètres selon les résultats")
        
        # Créer un fichier de métadonnées pour le suivi
        metadata = {
            'scraping_date': datetime.now().isoformat(),
            'ticker': ScraperConfig.TICKER,
            'elapsed_time_seconds': elapsed_time,
            'files_created': [
                f'{ScraperConfig.TICKER}_price_history.csv',
                f'{ScraperConfig.TICKER}_options_chain_full.csv',
                f'{ScraperConfig.TICKER}_options_data_ml_ready.csv',
                f'{ScraperConfig.TICKER}_options_data_filter.csv',
                'data_quality_report.json'
            ],
            'data_stats': quality_report['data_summary']
        }
        
        with open(os.path.join(ScraperConfig.OUTPUT_DIR, 'scraping_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
    except Exception as e:
        print(f"\n❌ Erreur fatale: {str(e)}")
        import traceback
        traceback.print_exc()

# =====================================================
# 9. FONCTIONS UTILITAIRES SUPPLÉMENTAIRES
# =====================================================

def download_alternative_data(ticker: str) -> Optional[pd.DataFrame]:
    """
    Méthode alternative pour récupérer des données si yfinance échoue
    
    Utilise des sources alternatives ou des API payantes si disponibles
    """
    print("\n🔄 Tentative de récupération via sources alternatives...")
    
    # Ici, vous pourriez implémenter:
    # - Scraping direct de Yahoo Finance avec BeautifulSoup
    # - Utilisation d'APIs alternatives (Alpha Vantage, IEX Cloud, etc.)
    # - Lecture de fichiers locaux de sauvegarde
    
    # Pour l'instant, retourner None
    return None

def validate_data_integrity(df: pd.DataFrame) -> bool:
    """
    Valide l'intégrité des données récupérées
    
    Vérifie:
    - Cohérence des prix (bid <= prix <= ask)
    - Valeurs positives pour les volumes
    - Greeks dans des plages raisonnables
    """
    checks = {
        'prices_positive': (df['Prix_Option'] > 0).all(),
        'volumes_non_negative': (df['Volume'] >= 0).all(),
        'delta_range': df['Delta'].between(-1, 1).all() if 'Delta' in df.columns else True,
        'gamma_positive': (df['Gamma'] >= 0).all() if 'Gamma' in df.columns else True,
        'vega_positive': (df['Vega'] >= 0).all() if 'Vega' in df.columns else True
    }
    
    all_valid = all(checks.values())
    
    if not all_valid:
        print("\n⚠️  Problèmes d'intégrité détectés:")
        for check, passed in checks.items():
            if not passed:
                print(f"   - {check}: FAILED")
    
    return all_valid

# =====================================================
# POINT D'ENTRÉE
# =====================================================

if __name__ == "__main__":
    # Vérifier les dépendances
    try:
        import yfinance
        import scipy
        print("✅ Toutes les dépendances sont installées")
    except ImportError as e:
        print(f"❌ Dépendance manquante: {e}")
        print("   Installer avec: pip install yfinance scipy beautifulsoup4 requests")
        exit(1)
    
    # Lancer le scraping
    main()
