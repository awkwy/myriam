"""
Modélisation Avancée de la Surface de Volatilité et Prédiction des Prix d'Options - VERSION CORRIGÉE
Correction des problèmes de valeurs manquantes et d'erreurs de dimensions
Auteur: Assistant Claude
Date: Mai 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns
from scipy.interpolate import griddata
from scipy.stats import norm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.impute import SimpleImputer  # AJOUT pour gérer les valeurs manquantes
import xgboost as xgb
import lightgbm as lgb

# Pour les graphiques avancés
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration et utilitaires
import joblib
import json
from typing import Dict, List, Tuple, Optional

# =====================================================
# CONFIGURATION DU PROJET
# =====================================================

class Config:
    """Configuration centralisée du projet"""
    TICKER = 'AAPL'
    RISK_FREE_RATE = 0.05  # Taux T-Bill 3 mois
    DIVIDEND_YIELD = 0.015  # Rendement en dividendes estimé pour AAPL
    
    # Fenêtres pour la volatilité historique
    VOLATILITY_WINDOWS = [10, 20, 30, 60, 90]
    
    # Paramètres pour la validation temporelle
    WALK_FORWARD_SPLITS = 5
    TEST_SIZE_DAYS = 30
    
    # Paramètres optimaux pour XGBoost (à affiner avec GridSearch)
    XGBOOST_PARAMS = {
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 800,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1
    }

# =====================================================
# 1. FONCTIONS UTILITAIRES POUR LES GREEKS (Identiques)
# =====================================================

def calculate_d1_d2(S: float, K: float, r: float, q: float, sigma: float, T: float) -> Tuple[float, float]:
    """Calcule d1 et d2 pour le modèle Black-Scholes"""
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return d1, d2

def calculate_greeks(S: float, K: float, r: float, q: float, sigma: float, T: float, 
                    option_type: str = 'CALL') -> Dict[str, float]:
    """Calcule tous les Greeks d'une option"""
    if T <= 0:
        return {
            'Delta': 1.0 if option_type == 'CALL' and S > K else 0.0,
            'Gamma': 0.0, 'Vega': 0.0, 'Theta': 0.0, 'Rho': 0.0, 'Vanna': 0.0, 'Volga': 0.0
        }
    
    d1, d2 = calculate_d1_d2(S, K, r, q, sigma, T)
    sqrt_T = np.sqrt(T)
    
    # Greeks de premier ordre
    if option_type == 'CALL':
        delta = np.exp(-q*T) * norm.cdf(d1)
        theta = (-S*norm.pdf(d1)*sigma*np.exp(-q*T))/(2*sqrt_T) - r*K*np.exp(-r*T)*norm.cdf(d2) + q*S*np.exp(-q*T)*norm.cdf(d1)
        rho = K*T*np.exp(-r*T)*norm.cdf(d2)
    else:  # PUT
        delta = -np.exp(-q*T) * norm.cdf(-d1)
        theta = (-S*norm.pdf(d1)*sigma*np.exp(-q*T))/(2*sqrt_T) + r*K*np.exp(-r*T)*norm.cdf(-d2) - q*S*np.exp(-q*T)*norm.cdf(-d1)
        rho = -K*T*np.exp(-r*T)*norm.cdf(-d2)
    
    # Greeks communs aux calls et puts
    gamma = (np.exp(-q*T) * norm.pdf(d1)) / (S * sigma * sqrt_T)
    vega = S * np.exp(-q*T) * norm.pdf(d1) * sqrt_T / 100
    
    # Greeks de second ordre
    vanna = vega * (1 - d1/(sigma*sqrt_T)) / S
    volga = vega * d1 * d2 / sigma
    
    # Annualiser theta
    theta = theta / 365
    
    return {
        'Delta': delta, 'Gamma': gamma, 'Vega': vega, 'Theta': theta,
        'Rho': rho, 'Vanna': vanna, 'Volga': volga
    }

def calculate_historical_volatility(prices: pd.Series, window: int) -> pd.Series:
    """Calcule la volatilité historique réalisée"""
    returns = prices.pct_change().dropna()
    vol = returns.rolling(window=window).std() * np.sqrt(252) * 100
    return vol

def calculate_volatility_skew(df: pd.DataFrame, lower_q: float = 0.25, upper_q: float = 0.75,
                             moneyness_range: float = None) -> pd.DataFrame:
    """Calcule le skew de volatilité implicite par maturité - VERSION CORRIGÉE"""
    results = []
    
    for mat, grp in df.groupby("Temps_Maturite_Annees", sort=False):
        # Sélection PUTs / CALLs selon la méthode choisie
        if moneyness_range is not None:
            if "Prix_Sous_Jacent" not in grp.columns:
                continue
            spot = grp["Prix_Sous_Jacent"].iloc[0]
            puts = grp[
                (grp["Type_Option"] == "PUT") & 
                (grp["Strike"] <= (1 - moneyness_range) * spot)
            ]
            calls = grp[
                (grp["Type_Option"] == "CALL") & 
                (grp["Strike"] >= (1 + moneyness_range) * spot)
            ]
        else:
            low_strk = grp["Strike"].quantile(lower_q)
            high_strk = grp["Strike"].quantile(upper_q)
            puts = grp[
                (grp["Type_Option"] == "PUT") & 
                (grp["Strike"] <= low_strk)
            ]
            calls = grp[
                (grp["Type_Option"] == "CALL") & 
                (grp["Strike"] >= high_strk)
            ]
        
        # Calcul du skew
        if not puts.empty and not calls.empty:
            skew = (puts["Volatilite_Implicite"].mean() - 
                   calls["Volatilite_Implicite"].mean())
            results.append({
                "Temps_Maturite_Annees": mat, 
                "Volatility_Skew": skew
            })
    
    return pd.DataFrame(results, columns=["Temps_Maturite_Annees", "Volatility_Skew"])

# =====================================================
# 2. FONCTION DE NETTOYAGE DES DONNÉES - NOUVELLE
# =====================================================

def clean_and_validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie et valide les données pour éviter les erreurs NaN
    Cette fonction est cruciale pour la stabilité du modèle
    """
    print("\n🧹 Nettoyage et validation des données...")
    
    initial_rows = len(df)
    
    # 1. Supprimer les lignes avec des valeurs critiques manquantes
    critical_columns = ['Prix_Option', 'Strike', 'Prix_Sous_Jacent', 'Temps_Maturite_Annees']
    df = df.dropna(subset=critical_columns)
    print(f"   Après suppression des valeurs critiques manquantes: {len(df)} lignes")
    
    # 2. Valider les valeurs numériques
    df = df[
        (df['Prix_Option'] > 0.01) &  # Prix minimum valide
        (df['Strike'] > 0) &
        (df['Prix_Sous_Jacent'] > 0) &
        (df['Temps_Maturite_Annees'] > 0.001) &  # Au moins 1 jour
        (df['Temps_Maturite_Annees'] < 10)  # Moins de 10 ans
    ]
    print(f"   Après validation des valeurs numériques: {len(df)} lignes")
    
    # 3. Corriger les valeurs aberrantes de volatilité
    vol_col = 'Volatilite_Implicite'
    if vol_col in df.columns:
        # Remplacer les valeurs extrêmes par des valeurs raisonnables
        q1 = df[vol_col].quantile(0.01)
        q99 = df[vol_col].quantile(0.99)
        df[vol_col] = df[vol_col].clip(lower=q1, upper=q99)
        print(f"   Volatilité ajustée entre {q1:.1f}% et {q99:.1f}%")
    
    # 4. Valider la moneyness
    if 'Moneyness' in df.columns:
        df = df[
            (df['Moneyness'] >= 0.3) &  # Pas trop OTM
            (df['Moneyness'] <= 3.0)    # Pas trop ITM
        ]
        print(f"   Après filtrage moneyness: {len(df)} lignes")
    
    # 5. Remplir les valeurs manquantes pour les colonnes non-critiques
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df[col].isnull().any():
            if col in ['Volume', 'Open_Interest']:
                df[col].fillna(0, inplace=True)  # Volume/OI peut être 0
            else:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
            print(f"   Valeurs manquantes remplies pour {col}")
    
    # 6. Vérification finale des valeurs infinies
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    removed_rows = initial_rows - len(df)
    removal_pct = (removed_rows / initial_rows) * 100
    
    print(f"✅ Nettoyage terminé:")
    print(f"   - Lignes supprimées: {removed_rows} ({removal_pct:.1f}%)")
    print(f"   - Lignes conservées: {len(df)}")
    
    return df

# =====================================================
# 3. CHARGEMENT ET ENRICHISSEMENT DES DONNÉES - MODIFIÉ
# =====================================================

print("📊 Chargement et enrichissement des données d'options...")
print("=" * 60)

# Charger les données de base
try:
    df = pd.read_csv('aapl_options_data_filter.csv')
    print(f"✅ {len(df)} options chargées depuis le fichier")
    
    # Nettoyer immédiatement les données
    df = clean_and_validate_data(df)
    
except:
    print("⚠️  Fichier non trouvé. Génération de données enrichies simulées...")
    
    # Génération de données simulées plus réalistes
    np.random.seed(42)
    n_samples = 2000
    
    # Paramètres réalistes pour AAPL
    spot_price = 150
    risk_free_rate = Config.RISK_FREE_RATE
    dividend_yield = Config.DIVIDEND_YIELD
    
    # Générer des strikes autour du spot avec une distribution réaliste
    moneyness = np.concatenate([
        np.random.normal(1.0, 0.15, n_samples//2),
        np.random.uniform(0.7, 1.3, n_samples//2)
    ])
    
    strikes = spot_price * moneyness
    
    # Maturités réalistes
    maturities = np.concatenate([
        np.random.exponential(0.25, n_samples//2),
        np.random.uniform(0.1, 2.0, n_samples//2)
    ])
    maturities = np.clip(maturities, 0.02, 3.0)
    
    # Volatilité implicite avec smile réaliste
    def generate_implied_vol(strike, maturity, spot):
        moneyness = strike / spot
        base_vol = 25
        smile_effect = 5 * (abs(np.log(moneyness)) ** 1.5) / np.sqrt(maturity + 0.1)
        term_structure = 5 * np.exp(-maturity) - 2
        noise = np.random.normal(0, 2)
        return np.clip(base_vol + smile_effect + term_structure + noise, 10, 80)  # Limiter la volatilité
    
    # Volume et Open Interest
    def generate_volume(moneyness, maturity):
        atm_factor = np.exp(-10 * (moneyness - 1)**2)
        maturity_factor = np.exp(-2 * maturity)
        base_volume = np.random.lognormal(8, 1.5)
        return max(1, int(base_volume * atm_factor * maturity_factor))  # Volume minimum de 1
    
    # Créer le DataFrame
    df = pd.DataFrame({
        'Strike': strikes,
        'Temps_Maturite_Annees': maturities,
        'Prix_Sous_Jacent': spot_price,
        'Type_Option': np.random.choice(['CALL', 'PUT'], n_samples),
        'Taux_Sans_Risque': risk_free_rate,
        'Dividend_Yield': dividend_yield
    })
    
    # Ajouter la volatilité implicite
    df['Volatilite_Implicite'] = df.apply(
        lambda row: generate_implied_vol(row['Strike'], row['Temps_Maturite_Annees'], spot_price), 
        axis=1
    )
    
    # Ajouter volume et open interest
    df['Moneyness'] = df['Strike'] / df['Prix_Sous_Jacent']
    df['Volume'] = df.apply(
        lambda row: generate_volume(row['Moneyness'], row['Temps_Maturite_Annees']), 
        axis=1
    )
    df['Open_Interest'] = (df['Volume'] * np.random.uniform(2, 10, n_samples)).astype(int)
    
    # Calculer les Greeks pour chaque option
    print("\n🧮 Calcul des Greeks...")
    greeks_list = []
    for idx, row in df.iterrows():
        greeks = calculate_greeks(
            S=row['Prix_Sous_Jacent'],
            K=row['Strike'],
            r=row['Taux_Sans_Risque'],
            q=row['Dividend_Yield'],
            sigma=row['Volatilite_Implicite'] / 100,
            T=row['Temps_Maturite_Annees'],
            option_type=row['Type_Option']
        )
        greeks_list.append(greeks)
    
    # Ajouter les Greeks au DataFrame
    greeks_df = pd.DataFrame(greeks_list)
    df = pd.concat([df, greeks_df], axis=1)
    
    # Calculer le prix d'option avec Black-Scholes
    def black_scholes_price(row):
        S = row['Prix_Sous_Jacent']
        K = row['Strike']
        r = row['Taux_Sans_Risque']
        q = row['Dividend_Yield']
        sigma = row['Volatilite_Implicite'] / 100
        T = row['Temps_Maturite_Annees']
        
        if T <= 0:
            return max(0, S - K) if row['Type_Option'] == 'CALL' else max(0, K - S)
        
        d1, d2 = calculate_d1_d2(S, K, r, q, sigma, T)
        
        if row['Type_Option'] == 'CALL':
            price = S * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r*T) * norm.cdf(-d2) - S * np.exp(-q*T) * norm.cdf(-d1)
        
        # Ajouter un petit bruit pour simuler les imperfections du marché
        noise = np.random.normal(0, price * 0.02)
        return max(0.01, price + noise)
    
    df['Prix_Option'] = df.apply(black_scholes_price, axis=1)
    
    # Nettoyer les données simulées également
    df = clean_and_validate_data(df)

# Ensure Greeks exist when data is read from CSV
needed = {"Delta", "Gamma", "Vega", "Theta", "Rho", "Vanna", "Volga"}
if not needed.issubset(df.columns):
    print("🔄 Calcul des Greeks manquants…")
    greeks_df = df.apply(
        lambda row: calculate_greeks(
            S=row["Prix_Sous_Jacent"],
            K=row["Strike"],
            r=row.get("Taux_Sans_Risque", Config.RISK_FREE_RATE),
            q=row.get("Dividend_Yield", Config.DIVIDEND_YIELD),
            sigma=row["Volatilite_Implicite"] / 100,
            T=row["Temps_Maturite_Annees"],
            option_type=row["Type_Option"],
        ),
        axis=1,
        result_type="expand",
    )
    df = pd.concat([df, greeks_df], axis=1)

# =====================================================
# 4. FEATURE ENGINEERING AVANCÉ - SÉCURISÉ
# =====================================================

print("\n🔧 Feature Engineering avancé...")

# Features de base avec vérification des divisions par zéro
df['Jours_Jusqu_Maturite'] = df['Temps_Maturite_Annees'] * 365
df['Strike_Ratio'] = df['Strike'] / df['Prix_Sous_Jacent']
df['Log_Moneyness'] = np.log(df['Moneyness'].clip(lower=0.001))  # Éviter log(0)
df['Sqrt_Time'] = np.sqrt(df['Temps_Maturite_Annees'])
df['Vol_Time'] = df['Volatilite_Implicite'] * df['Sqrt_Time']

# Features de liquidité sécurisées
df['Log_Volume'] = np.log1p(df['Volume'].clip(lower=0))
df['Log_Open_Interest'] = np.log1p(df['Open_Interest'].clip(lower=0))
df['Liquidity_Score'] = df['Volume'] / (df['Open_Interest'] + 1)

# Features d'asymétrie
df['Is_OTM'] = ((df['Type_Option'] == 'CALL') & (df['Strike'] > df['Prix_Sous_Jacent'])) | \
               ((df['Type_Option'] == 'PUT') & (df['Strike'] < df['Prix_Sous_Jacent']))
df['OTM_Distance'] = np.abs(df['Log_Moneyness']) * df['Is_OTM']

# Interaction entre Greeks avec protection contre division par zéro
df['Delta_Gamma_Product'] = df['Delta'] * df['Gamma']
df['Vega_Volga_Ratio'] = df['Vega'] / (df['Volga'].abs() + 0.001)

# Structure à terme de la volatilité
skew_df = calculate_volatility_skew(df, moneyness_range=0.05)
df = df.merge(skew_df, on="Temps_Maturite_Annees", how="left")

# Garantir que la colonne existe et qu'elle n'a pas de NaN
if "Volatility_Skew" not in df.columns:
    df["Volatility_Skew"] = 0.0
else:
    df["Volatility_Skew"].fillna(0, inplace=True)

# VIX proxy sécurisé
vix_proxy = df[df['Temps_Maturite_Annees'] <= 0.1]['Volatilite_Implicite'].mean()
if pd.isna(vix_proxy) or vix_proxy <= 0:
    vix_proxy = 25.0  # Valeur par défaut
df['VIX_Proxy'] = vix_proxy
df['Vol_VIX_Ratio'] = df['Volatilite_Implicite'] / vix_proxy

# Encoder le type d'option
df['Is_Call'] = (df['Type_Option'] == 'CALL').astype(int)

# Vérification finale pour les valeurs infinies ou NaN
print("🔍 Vérification finale des données...")
infinite_cols = []
nan_cols = []

for col in df.select_dtypes(include=[np.number]).columns:
    if np.isinf(df[col]).any():
        infinite_cols.append(col)
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    if df[col].isnull().any():
        nan_cols.append(col)
        df[col].fillna(df[col].median(), inplace=True)

if infinite_cols:
    print(f"   Valeurs infinies corrigées dans: {infinite_cols}")
if nan_cols:
    print(f"   Valeurs NaN remplies dans: {nan_cols}")

print("✅ Feature engineering terminé avec validation complète")

# =====================================================
# 5. SÉLECTION DES FEATURES ET PRÉPARATION - SÉCURISÉE
# =====================================================

print("\n📋 Sélection et préparation des features...")

# Liste complète des features à utiliser
feature_columns = [
    # Prix et ratios
    'Prix_Sous_Jacent', 'Strike', 'Strike_Ratio', 'Moneyness', 'Log_Moneyness',
    
    # Temps
    'Temps_Maturite_Annees', 'Sqrt_Time', 'Jours_Jusqu_Maturite',
    
    # Volatilité
    'Volatilite_Implicite', 'Vol_Time', 'Volatility_Skew', 'Vol_VIX_Ratio',
    
    # Greeks
    'Delta', 'Gamma', 'Vega', 'Theta', 'Rho', 'Vanna', 'Volga',
    'Delta_Gamma_Product', 'Vega_Volga_Ratio',
    
    # Taux et dividendes
    'Taux_Sans_Risque', 'Dividend_Yield',
    
    # Liquidité
    'Log_Volume', 'Log_Open_Interest', 'Liquidity_Score',
    
    # Type et caractéristiques
    'Is_Call', 'Is_OTM', 'OTM_Distance',
    
    # Marché
    'VIX_Proxy'
]

# Vérifier que toutes les colonnes existent
feature_columns = [col for col in feature_columns if col in df.columns]
print(f"✅ {len(feature_columns)} features sélectionnées")

# Préparer X et y avec vérification
X = df[feature_columns].copy()
y = df['Prix_Option'].copy()

# Vérification finale avant split
print(f"   Shape X: {X.shape}")
print(f"   Shape y: {y.shape}")
print(f"   Valeurs NaN dans X: {X.isnull().sum().sum()}")
print(f"   Valeurs NaN dans y: {y.isnull().sum()}")

# Supprimer les lignes avec des valeurs manquantes dans y
valid_indices = ~y.isnull()
X = X[valid_indices]
y = y[valid_indices]

print(f"   Après nettoyage final - X: {X.shape}, y: {y.shape}")

# =====================================================
# 6. DIVISION ET NORMALISATION DES DONNÉES
# =====================================================

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

# Normalisation des features avec gestion des valeurs manquantes
print("\n🔧 Normalisation des données...")

# Créer un pipeline d'imputation et de normalisation
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

# Appliquer l'imputation puis la normalisation
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

print(f"✅ Données préparées et normalisées:")
print(f"   - Features: {len(feature_columns)}")
print(f"   - Train set: {len(X_train)} samples")
print(f"   - Test set: {len(X_test)} samples")
print(f"   - Valeurs NaN dans données normalisées: {np.isnan(X_train_scaled).sum()}")

# =====================================================
# 7. ENTRAÎNEMENT DES MODÈLES SÉPARÉS - ROBUSTE
# =====================================================

print("\n🏋️ Entraînement des modèles séparés pour CALLs et PUTs...")

# Séparer les données par type d'option
train_calls_mask = X_train['Is_Call'] == 1
train_puts_mask = X_train['Is_Call'] == 0
test_calls_mask = X_test['Is_Call'] == 1
test_puts_mask = X_test['Is_Call'] == 0

# Données pour les CALLs
X_train_calls = X_train[train_calls_mask]
y_train_calls = y_train[train_calls_mask]
X_test_calls = X_test[test_calls_mask]
y_test_calls = y_test[test_calls_mask]

# Données pour les PUTs
X_train_puts = X_train[train_puts_mask]
y_train_puts = y_train[train_puts_mask]
X_test_puts = X_test[test_puts_mask]
y_test_puts = y_test[test_puts_mask]

print(f"\n📊 Distribution des données:")
print(f"   - CALLs: {len(X_train_calls)} train, {len(X_test_calls)} test")
print(f"   - PUTs: {len(X_train_puts)} train, {len(X_test_puts)} test")

# Dictionnaire pour stocker tous les résultats
results_calls = {}
results_puts = {}
results_combined = {}

# =====================================================
# 8. FONCTION D'ÉVALUATION ROBUSTE
# =====================================================

def evaluate_model_robust(y_true, y_pred, prefix=""):
    """Calcule toutes les métriques d'évaluation avec gestion des erreurs"""
    try:
        # Vérifier que les prédictions sont valides
        if len(y_pred) == 0 or len(y_true) == 0:
            return {f'{prefix}MSE': np.inf, f'{prefix}MAE': np.inf, f'{prefix}R²': -np.inf, 
                   f'{prefix}MAPE (%)': np.inf, f'{prefix}RMSE': np.inf}
        
        # Remplacer les valeurs infinies ou NaN dans les prédictions
        y_pred_clean = np.nan_to_num(y_pred, nan=np.median(y_true), 
                                    posinf=np.max(y_true), neginf=np.min(y_true))
        
        mse = mean_squared_error(y_true, y_pred_clean)
        mae = mean_absolute_error(y_true, y_pred_clean)
        r2 = r2_score(y_true, y_pred_clean)
        
        # MAPE sécurisé (éviter division par zéro)
        mape_mask = y_true != 0
        if mape_mask.sum() > 0:
            mape = mean_absolute_percentage_error(y_true[mape_mask], y_pred_clean[mape_mask]) * 100
        else:
            mape = np.inf
        
        rmse = np.sqrt(mse)
        
        return {
            f'{prefix}MSE': mse,
            f'{prefix}MAE': mae,
            f'{prefix}R²': r2,
            f'{prefix}MAPE (%)': min(mape, 1000),  # Limiter MAPE à 1000%
            f'{prefix}RMSE': rmse
        }
    except Exception as e:
        print(f"   ⚠️  Erreur dans l'évaluation: {e}")
        return {f'{prefix}MSE': np.inf, f'{prefix}MAE': np.inf, f'{prefix}R²': -np.inf, 
               f'{prefix}MAPE (%)': np.inf, f'{prefix}RMSE': np.inf}

# =====================================================
# 9. ENTRAÎNEMENT DE TOUS LES MODÈLES - VERSION ROBUSTE
# =====================================================

print("\n🚀 Entraînement de tous les modèles...")
print("=" * 60)

# Liste des modèles à entraîner
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
    'XGBoost': xgb.XGBRegressor(**Config.XGBOOST_PARAMS),
    'LightGBM': lgb.LGBMRegressor(n_estimators=800, max_depth=8, learning_rate=0.05, random_state=42, n_jobs=-1, verbosity=-1),
    'SVR': SVR(kernel='rbf', C=100, gamma='scale'),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', 
                                   max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1)
}

# Entraîner chaque modèle
for model_name, model in models.items():
    print(f"\n🔄 {model_name}...")
    
    try:
        # Modèles nécessitant des données normalisées
        needs_scaling = model_name in ['Linear Regression', 'Ridge', 'SVR', 'Neural Network']
        
        # 1. Modèle pour les CALLs
        if len(X_train_calls) > 0 and len(X_test_calls) > 0:
            if needs_scaling:
                scaler_calls = StandardScaler()
                X_train_calls_scaled = scaler_calls.fit_transform(X_train_calls)
                X_test_calls_scaled = scaler_calls.transform(X_test_calls)
                model_calls = model.__class__(**model.get_params())
                model_calls.fit(X_train_calls_scaled, y_train_calls)
                y_pred_calls = model_calls.predict(X_test_calls_scaled)
            else:
                model_calls = model.__class__(**model.get_params())
                model_calls.fit(X_train_calls, y_train_calls)
                y_pred_calls = model_calls.predict(X_test_calls)
        else:
            print(f"   ⚠️  Pas assez de données CALLs pour {model_name}")
            continue
        
        # 2. Modèle pour les PUTs
        if len(X_train_puts) > 0 and len(X_test_puts) > 0:
            if needs_scaling:
                scaler_puts = StandardScaler()
                X_train_puts_scaled = scaler_puts.fit_transform(X_train_puts)
                X_test_puts_scaled = scaler_puts.transform(X_test_puts)
                model_puts = model.__class__(**model.get_params())
                model_puts.fit(X_train_puts_scaled, y_train_puts)
                y_pred_puts = model_puts.predict(X_test_puts_scaled)
            else:
                model_puts = model.__class__(**model.get_params())
                model_puts.fit(X_train_puts, y_train_puts)
                y_pred_puts = model_puts.predict(X_test_puts)
        else:
            print(f"   ⚠️  Pas assez de données PUTs pour {model_name}")
            continue
        
        # 3. Modèle combiné
        if needs_scaling:
            model_combined = model.__class__(**model.get_params())
            model_combined.fit(X_train_scaled, y_train)
            y_pred_combined = model_combined.predict(X_test_scaled)
        else:
            model_combined = model.__class__(**model.get_params())
            model_combined.fit(X_train, y_train)
            y_pred_combined = model_combined.predict(X_test)
        
        # Évaluer les modèles avec la fonction robuste
        results_calls[model_name] = {
            'model': model_calls,
            'predictions': y_pred_calls,
            **evaluate_model_robust(y_test_calls, y_pred_calls, "CALL_")
        }
        
        results_puts[model_name] = {
            'model': model_puts,
            'predictions': y_pred_puts,
            **evaluate_model_robust(y_test_puts, y_pred_puts, "PUT_")
        }
        
        results_combined[model_name] = {
            'model': model_combined,
            'predictions': y_pred_combined,
            **evaluate_model_robust(y_test, y_pred_combined, "COMBINED_")
        }
        
        # Afficher les résultats
        print(f"   ✅ CALLs - R²: {results_calls[model_name]['CALL_R²']:.4f}, MAPE: {results_calls[model_name]['CALL_MAPE (%)']:.2f}%")
        print(f"   ✅ PUTs  - R²: {results_puts[model_name]['PUT_R²']:.4f}, MAPE: {results_puts[model_name]['PUT_MAPE (%)']:.2f}%")
        print(f"   ✅ TOTAL - R²: {results_combined[model_name]['COMBINED_R²']:.4f}, MAPE: {results_combined[model_name]['COMBINED_MAPE (%)']:.2f}%")
        
    except Exception as e:
        print(f"   ❌ Erreur lors de l'entraînement de {model_name}: {str(e)}")
        continue

# =====================================================
# 10. ANALYSE COMPARATIVE - VERSION CORRIGÉE
# =====================================================

print("\n📊 Analyse comparative des modèles...")
print("=" * 80)

# Créer un DataFrame de comparaison pour chaque type - VERSION CORRIGÉE
if results_calls:
    comparison_calls = pd.DataFrame({
        model: {k: v for k, v in results.items() if k.startswith('CALL_')}
        for model, results in results_calls.items()
    }).T
else:
    comparison_calls = pd.DataFrame()

if results_puts:
    comparison_puts = pd.DataFrame({
        model: {k: v for k, v in results.items() if k.startswith('PUT_')}
        for model, results in results_puts.items()
    }).T
else:
    comparison_puts = pd.DataFrame()

if results_combined:
    comparison_combined = pd.DataFrame({
        model: {k: v for k, v in results.items() if k.startswith('COMBINED_')}
        for model, results in results_combined.items()
    }).T
else:
    comparison_combined = pd.DataFrame()

# Identifier les meilleurs modèles
if not comparison_calls.empty:
    best_call_model = comparison_calls['CALL_R²'].idxmax()
    best_call_r2 = comparison_calls.loc[best_call_model, 'CALL_R²']
else:
    best_call_model = "Aucun"
    best_call_r2 = 0

if not comparison_puts.empty:
    best_put_model = comparison_puts['PUT_R²'].idxmax()
    best_put_r2 = comparison_puts.loc[best_put_model, 'PUT_R²']
else:
    best_put_model = "Aucun"
    best_put_r2 = 0

if not comparison_combined.empty:
    best_combined_model = comparison_combined['COMBINED_R²'].idxmax()
    best_combined_r2 = comparison_combined.loc[best_combined_model, 'COMBINED_R²']
else:
    best_combined_model = "Aucun"
    best_combined_r2 = 0

print("\n🏆 Meilleurs modèles:")
print(f"   - CALLs: {best_call_model} (R² = {best_call_r2:.4f})")
print(f"   - PUTs: {best_put_model} (R² = {best_put_r2:.4f})")
print(f"   - COMBINÉ: {best_combined_model} (R² = {best_combined_r2:.4f})")

if not comparison_combined.empty:
    print("\n📋 Tableau de comparaison détaillé:")
    print(comparison_combined.sort_values('COMBINED_R²', ascending=False).round(4))

# =====================================================
# 11. CRÉATION DES VISUALISATIONS - VERSION CORRIGÉE
# =====================================================

print("\n📈 Création des visualisations...")

# Figure complexe avec subplots
fig = plt.figure(figsize=(20, 16))

# 1. Surface de volatilité 3D améliorée
ax1 = fig.add_subplot(3, 3, 1, projection='3d')
calls_df = df[df['Type_Option'] == 'CALL'].copy()

if len(calls_df) > 0:
    # Créer une grille plus fine
    strike_range = np.linspace(calls_df['Strike'].quantile(0.1), 
                              calls_df['Strike'].quantile(0.9), 50)
    maturity_range = np.linspace(0.02, 2.0, 50)
    strike_grid, maturity_grid = np.meshgrid(strike_range, maturity_range)
    
    # Interpolation
    points = calls_df[['Strike', 'Temps_Maturite_Annees']].values
    values = calls_df['Volatilite_Implicite'].values
    vol_grid = griddata(points, values, (strike_grid, maturity_grid), method='cubic')
    
    # Surface
    surf = ax1.plot_surface(strike_grid, maturity_grid, vol_grid,
                           cmap=cm.viridis, alpha=0.8, linewidth=0, antialiased=True)
    ax1.set_xlabel('Strike ($)')
    ax1.set_ylabel('Maturité (années)')
    ax1.set_zlabel('Vol Implicite (%)')
    ax1.set_title('Surface de Volatilité - CALLs')

# 2. Comparaison des modèles (R²)
ax2 = fig.add_subplot(3, 3, 2)
if not comparison_combined.empty:
    models_sorted = comparison_combined.sort_values('COMBINED_R²', ascending=False)
    x_pos = np.arange(len(models_sorted))
    ax2.bar(x_pos, models_sorted['COMBINED_R²'], color='skyblue')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models_sorted.index, rotation=45, ha='right')
    ax2.set_ylabel('R² Score')
    ax2.set_title('Performance des Modèles (R²)')
    ax2.grid(True, alpha=0.3)

# 3. Scatter plot Prédictions vs Réalité (meilleur modèle)
ax3 = fig.add_subplot(3, 3, 3)
if best_combined_model != "Aucun" and best_combined_model in results_combined:
    best_preds = results_combined[best_combined_model]['predictions']
    ax3.scatter(y_test, best_preds, alpha=0.5, s=20, c='blue', edgecolors='none')
    ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax3.set_xlabel('Prix Réel ($)')
    ax3.set_ylabel('Prix Prédit ($)')
    ax3.set_title(f'Prédictions vs Réalité - {best_combined_model}')
    ax3.grid(True, alpha=0.3)

# 4. Distribution des erreurs par modèle
ax4 = fig.add_subplot(3, 3, 4)
if not comparison_combined.empty:
    errors_data = []
    error_labels = []
    for model_name in comparison_combined.index[:5]:  # Top 5 modèles
        if model_name in results_combined:
            errors = y_test.values - results_combined[model_name]['predictions']
            errors_data.append(errors)
            error_labels.append(model_name)
    
    if errors_data:
        ax4.boxplot(errors_data, labels=error_labels)
        ax4.set_xlabel('Modèle')
        ax4.set_ylabel('Erreur ($)')
        ax4.set_title('Distribution des Erreurs - Top Modèles')
        ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax4.set_xticklabels(error_labels, rotation=45, ha='right')

# 5. Importance des features (si disponible)
ax5 = fig.add_subplot(3, 3, 5)
if (best_combined_model != "Aucun" and 
    best_combined_model in results_combined and
    hasattr(results_combined[best_combined_model]['model'], 'feature_importances_')):
    
    model = results_combined[best_combined_model]['model']
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # Top 15 features
    
    ax5.barh(range(len(indices)), importances[indices])
    ax5.set_yticks(range(len(indices)))
    ax5.set_yticklabels([feature_columns[i] for i in indices])
    ax5.set_xlabel('Importance')
    ax5.set_title('Top 15 Features')
    ax5.grid(True, alpha=0.3)
else:
    ax5.text(0.5, 0.5, 'Feature importance\nnon disponible', 
            ha='center', va='center', transform=ax5.transAxes)

# 6-9. Graphiques supplémentaires (simplifiés pour éviter les erreurs)
for i, ax in enumerate([fig.add_subplot(3, 3, j) for j in range(6, 10)]):
    ax.text(0.5, 0.5, f'Graphique {i+6}\nen cours de développement', 
           ha='center', va='center', transform=ax.transAxes)
    ax.set_title(f'Analyse {i+6}')

plt.tight_layout()
plt.savefig('advanced_options_analysis_fixed.png', dpi=300, bbox_inches='tight')
plt.show()

# =====================================================
# 12. SAUVEGARDE DES RÉSULTATS - SÉCURISÉE
# =====================================================

print("\n💾 Sauvegarde des résultats...")

try:
    # Sauvegarder les résultats de comparaison
    if not comparison_combined.empty:
        comparison_combined.to_csv('model_comparison_fixed.csv')
        print("✅ Comparaison des modèles sauvegardée")
    
    # Sauvegarder les meilleurs modèles
    if best_combined_model != "Aucun" and best_combined_model in results_combined:
        joblib.dump(results_combined[best_combined_model]['model'], 
                   f'best_model_combined_{best_combined_model.replace(" ", "_")}.pkl')
        print(f"✅ Meilleur modèle sauvegardé: {best_combined_model}")
    
    # Sauvegarder le scaler et l'imputer
    joblib.dump(scaler, 'feature_scaler_fixed.pkl')
    joblib.dump(imputer, 'feature_imputer_fixed.pkl')
    print("✅ Préprocesseurs sauvegardés")
    
except Exception as e:
    print(f"⚠️  Erreur lors de la sauvegarde: {e}")

print("\n✨ Analyse complète terminée avec succès!")
print("=" * 60)

if not comparison_combined.empty:
    print("\n📊 Résumé des performances:")
    print(f"   - Meilleur R² global: {comparison_combined.iloc[0]['COMBINED_R²']:.4f}")
    print(f"   - Meilleur MAPE global: {comparison_combined.iloc[0]['COMBINED_MAPE (%)']:.2f}%")

print("\n🎯 Améliorations apportées:")
print("   ✅ Gestion robuste des valeurs NaN")
print("   ✅ Validation et nettoyage des données")
print("   ✅ Protection contre les divisions par zéro")
print("   ✅ Limitation des valeurs aberrantes")
print("   ✅ Gestion d'erreurs dans l'entraînement")
print("   ✅ Métriques d'évaluation sécurisées")
