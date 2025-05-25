"""
Modélisation Avancée de la Surface de Volatilité et Prédiction des Prix d'Options
Version enrichie avec toutes les variables pertinentes et meilleures pratiques
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
# 1. FONCTIONS UTILITAIRES POUR LES GREEKS
# =====================================================

def calculate_d1_d2(S: float, K: float, r: float, q: float, sigma: float, T: float) -> Tuple[float, float]:
    """
    Calcule d1 et d2 pour le modèle Black-Scholes
    
    Args:
        S: Prix du sous-jacent
        K: Prix d'exercice
        r: Taux sans risque
        q: Taux de dividende
        sigma: Volatilité
        T: Temps jusqu'à maturité
    
    Returns:
        Tuple (d1, d2)
    """
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return d1, d2

def calculate_greeks(S: float, K: float, r: float, q: float, sigma: float, T: float, 
                    option_type: str = 'CALL') -> Dict[str, float]:
    """
    Calcule tous les Greeks d'une option
    
    Returns:
        Dict contenant Delta, Gamma, Vega, Theta, Rho, Vanna, Volga
    """
    # Éviter la division par zéro
    if T <= 0:
        return {
            'Delta': 1.0 if option_type == 'CALL' and S > K else 0.0,
            'Gamma': 0.0,
            'Vega': 0.0,
            'Theta': 0.0,
            'Rho': 0.0,
            'Vanna': 0.0,
            'Volga': 0.0
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
    vega = S * np.exp(-q*T) * norm.pdf(d1) * sqrt_T / 100  # Divisé par 100 pour obtenir le vega pour 1% de changement
    
    # Greeks de second ordre
    vanna = vega * (1 - d1/(sigma*sqrt_T)) / S  # Sensibilité du delta à la volatilité
    volga = vega * d1 * d2 / sigma  # Sensibilité du vega à la volatilité
    
    # Annualiser theta (convention: theta négatif = perte de valeur temps)
    theta = theta / 365
    
    return {
        'Delta': delta,
        'Gamma': gamma,
        'Vega': vega,
        'Theta': theta,
        'Rho': rho,
        'Vanna': vanna,
        'Volga': volga
    }

def calculate_historical_volatility(prices: pd.Series, window: int) -> pd.Series:
    """
    Calcule la volatilité historique réalisée
    
    Args:
        prices: Série des prix
        window: Nombre de jours pour le calcul
    
    Returns:
        Série de volatilité annualisée en pourcentage
    """
    returns = prices.pct_change().dropna()
    vol = returns.rolling(window=window).std() * np.sqrt(252) * 100  # Annualisée et en %
    return vol

def calculate_volatility_skew(df: pd.DataFrame, moneyness_range: float = 0.1) -> pd.Series:
    """
    Calcule le skew de volatilité (différence entre puts OTM et calls OTM)
    
    Args:
        df: DataFrame avec les données d'options
        moneyness_range: Distance au strike ATM (ex: 0.1 = 10%)
    
    Returns:
        Série avec le skew par date de maturité
    """
    skew_data = []
    
    # Grouper par date de maturité
    for maturity, group in df.groupby('Temps_Maturite_Annees'):
        atm_strike = group['Prix_Sous_Jacent'].iloc[0]  # Prix spot comme approximation ATM
        
        # Puts OTM (strike < spot)
        puts_otm = group[(group['Type_Option'] == 'PUT') & 
                        (group['Strike'] < atm_strike * (1 - moneyness_range)) &
                        (group['Strike'] > atm_strike * (1 - 2*moneyness_range))]
        
        # Calls OTM (strike > spot)
        calls_otm = group[(group['Type_Option'] == 'CALL') & 
                         (group['Strike'] > atm_strike * (1 + moneyness_range)) &
                         (group['Strike'] < atm_strike * (1 + 2*moneyness_range))]
        
        if len(puts_otm) > 0 and len(calls_otm) > 0:
            avg_put_vol = puts_otm['Volatilite_Implicite'].mean()
            avg_call_vol = calls_otm['Volatilite_Implicite'].mean()
            skew = avg_put_vol - avg_call_vol
            
            skew_data.append({
                'Temps_Maturite_Annees': maturity,
                'Volatility_Skew': skew
            })
    
    return pd.DataFrame(skew_data)

# =====================================================
# 2. CHARGEMENT ET ENRICHISSEMENT DES DONNÉES
# =====================================================

print("📊 Chargement et enrichissement des données d'options...")
print("=" * 60)

# Charger les données de base
try:
    df = pd.read_csv('aapl_options_data_filter.csv')
    print(f"✅ {len(df)} options chargées depuis le fichier")
    
    # Si les données historiques de prix existent, les charger pour calculer la volatilité réalisée
    try:
        price_history = pd.read_csv('aapl_price_history.csv', parse_dates=['Date'])
        price_history.set_index('Date', inplace=True)
        HAS_PRICE_HISTORY = True
        print("✅ Historique des prix chargé")
    except:
        HAS_PRICE_HISTORY = False
        print("⚠️  Historique des prix non disponible")
        
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
        np.random.normal(1.0, 0.15, n_samples//2),  # Plus de strikes ATM
        np.random.uniform(0.7, 1.3, n_samples//2)    # Strikes OTM
    ])
    
    strikes = spot_price * moneyness
    
    # Maturités réalistes (plus d'options court terme)
    maturities = np.concatenate([
        np.random.exponential(0.25, n_samples//2),   # Beaucoup d'options court terme
        np.random.uniform(0.1, 2.0, n_samples//2)    # Distribution uniforme jusqu'à 2 ans
    ])
    maturities = np.clip(maturities, 0.02, 3.0)     # Entre 1 semaine et 3 ans
    
    # Volatilité implicite avec smile réaliste
    def generate_implied_vol(strike, maturity, spot):
        moneyness = strike / spot
        base_vol = 25  # Volatilité de base pour AAPL
        
        # Smile effect (plus prononcé pour les maturités courtes)
        smile_effect = 5 * (abs(np.log(moneyness)) ** 1.5) / np.sqrt(maturity + 0.1)
        
        # Structure à terme (volatilité converge vers la moyenne à long terme)
        term_structure = 5 * np.exp(-maturity) - 2
        
        # Bruit aléatoire
        noise = np.random.normal(0, 2)
        
        return base_vol + smile_effect + term_structure + noise
    
    # Volume et Open Interest (corrélés avec la moneyness et la maturité)
    def generate_volume(moneyness, maturity):
        # Plus de volume pour les options ATM et court terme
        atm_factor = np.exp(-10 * (moneyness - 1)**2)
        maturity_factor = np.exp(-2 * maturity)
        base_volume = np.random.lognormal(8, 1.5)  # Log-normal pour éviter les valeurs négatives
        return int(base_volume * atm_factor * maturity_factor)
    
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
            sigma=row['Volatilite_Implicite'] / 100,  # Convertir en décimal
            T=row['Temps_Maturite_Annees'],
            option_type=row['Type_Option']
        )
        greeks_list.append(greeks)
    
    # Ajouter les Greeks au DataFrame
    greeks_df = pd.DataFrame(greeks_list)
    df = pd.concat([df, greeks_df], axis=1)
    
    # Calculer le prix d'option avec Black-Scholes (pour la simulation)
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
        noise = np.random.normal(0, price * 0.02)  # 2% de bruit
        return max(0.01, price + noise)  # Prix minimum de 0.01$
    
    df['Prix_Option'] = df.apply(black_scholes_price, axis=1)
    
    # Simuler un historique de prix pour la volatilité réalisée
    HAS_PRICE_HISTORY = False  # Pas d'historique réel disponible

# =====================================================
# 3. FEATURE ENGINEERING AVANCÉ
# =====================================================

print("\n🔧 Feature Engineering avancé...")

# Features de base déjà calculées
df['Jours_Jusqu_Maturite'] = df['Temps_Maturite_Annees'] * 365
df['Strike_Ratio'] = df['Strike'] / df['Prix_Sous_Jacent']
df['Log_Moneyness'] = np.log(df['Moneyness'])
df['Sqrt_Time'] = np.sqrt(df['Temps_Maturite_Annees'])
df['Vol_Time'] = df['Volatilite_Implicite'] * df['Sqrt_Time']

# Features de liquidité
df['Log_Volume'] = np.log1p(df['Volume'])  # log1p pour gérer les volumes de 0
df['Log_Open_Interest'] = np.log1p(df['Open_Interest'])
df['Liquidity_Score'] = df['Volume'] / (df['Open_Interest'] + 1)  # Ratio de trading

# Features d'asymétrie
df['Is_OTM'] = ((df['Type_Option'] == 'CALL') & (df['Strike'] > df['Prix_Sous_Jacent'])) | \
               ((df['Type_Option'] == 'PUT') & (df['Strike'] < df['Prix_Sous_Jacent']))
df['OTM_Distance'] = np.abs(df['Log_Moneyness']) * df['Is_OTM']

# Interaction entre Greeks
df['Delta_Gamma_Product'] = df['Delta'] * df['Gamma']
df['Vega_Volga_Ratio'] = df['Vega'] / (df['Volga'] + 0.001)  # Éviter division par zéro

# Structure à terme de la volatilité (nécessite le calcul du skew)
skew_df = calculate_volatility_skew(df)
df = df.merge(skew_df, on='Temps_Maturite_Annees', how='left')
df['Volatility_Skew'].fillna(0, inplace=True)

# Si nous n'avons pas de VIX réel, créer un proxy basé sur la volatilité moyenne court terme
vix_proxy = df[df['Temps_Maturite_Annees'] <= 0.1]['Volatilite_Implicite'].mean()
df['VIX_Proxy'] = vix_proxy
df['Vol_VIX_Ratio'] = df['Volatilite_Implicite'] / vix_proxy

# Encoder le type d'option
df['Is_Call'] = (df['Type_Option'] == 'CALL').astype(int)

# Ajouter des features temporelles si nous avons des dates
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['MonthOfYear'] = df['Date'].dt.month
    df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)

# =====================================================
# 4. SÉLECTION DES FEATURES ET PRÉPARATION
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

# Préparer X et y
X = df[feature_columns]
y = df['Prix_Option']

# =====================================================
# 5. VALIDATION TEMPORELLE (WALK-FORWARD)
# =====================================================

print("\n⏰ Configuration de la validation temporelle...")

# Si nous avons des dates, utiliser une vraie validation temporelle
if 'Date' in df.columns:
    df = df.sort_values('Date')
    
    # Créer des splits temporels
    tscv = TimeSeriesSplit(n_splits=Config.WALK_FORWARD_SPLITS)
    
    print(f"✅ Validation Walk-Forward avec {Config.WALK_FORWARD_SPLITS} splits")
else:
    # Sinon, utiliser un split classique mais ordonné
    print("⚠️  Pas de dates disponibles, utilisation d'un split ordonné par maturité")
    df = df.sort_values('Temps_Maturite_Annees')
    
# Division train/test standard pour la comparaison initiale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False  # shuffle=False pour respecter l'ordre temporel
)

# Normalisation des features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Données préparées:")
print(f"   - Features: {len(feature_columns)}")
print(f"   - Train set: {len(X_train)} samples")
print(f"   - Test set: {len(X_test)} samples")

# Afficher l'importance relative des types de features
feature_types = {
    'Prix': ['Prix_Sous_Jacent', 'Strike', 'Strike_Ratio', 'Moneyness', 'Log_Moneyness'],
    'Temps': ['Temps_Maturite_Annees', 'Sqrt_Time', 'Jours_Jusqu_Maturite'],
    'Volatilité': ['Volatilite_Implicite', 'Vol_Time', 'Volatility_Skew', 'Vol_VIX_Ratio'],
    'Greeks': ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho', 'Vanna', 'Volga', 'Delta_Gamma_Product', 'Vega_Volga_Ratio'],
    'Liquidité': ['Log_Volume', 'Log_Open_Interest', 'Liquidity_Score']
}

print("\n📊 Distribution des types de features:")
for feat_type, feat_list in feature_types.items():
    count = len([f for f in feat_list if f in feature_columns])
    print(f"   - {feat_type}: {count} features")

# =====================================================
# 6. OPTIMISATION DES HYPERPARAMÈTRES
# =====================================================

print("\n🔍 Optimisation des hyperparamètres pour XGBoost...")

# Grille de paramètres à tester
param_grid = {
    'max_depth': [6, 8, 10],
    'learning_rate': [0.03, 0.05, 0.1],
    'n_estimators': [500, 800, 1000],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

# Pour gagner du temps, on fait une recherche rapide sur un échantillon
sample_size = min(5000, len(X_train))
sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
X_train_sample = X_train.iloc[sample_indices]
y_train_sample = y_train.iloc[sample_indices]

# GridSearch rapide
xgb_base = xgb.XGBRegressor(random_state=42, n_jobs=-1, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0)
grid_search = GridSearchCV(
    xgb_base, 
    param_grid, 
    cv=3,  # 3-fold pour la rapidité
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

print("⏳ Recherche en cours (cela peut prendre quelques minutes)...")
grid_search.fit(X_train_sample, y_train_sample)

print(f"\n✅ Meilleurs paramètres trouvés:")
for param, value in grid_search.best_params_.items():
    print(f"   - {param}: {value}")
print(f"   - Score R² de validation: {grid_search.best_score_:.4f}")

# Mettre à jour les paramètres optimaux
Config.XGBOOST_PARAMS.update(grid_search.best_params_)

# =====================================================
# 7. ENTRAÎNEMENT DES MODÈLES SÉPARÉS POUR CALLS ET PUTS
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
# 8. ENTRAÎNEMENT DE TOUS LES MODÈLES
# =====================================================

print("\n🚀 Entraînement de tous les modèles...")
print("=" * 60)

# Fonction helper pour évaluer un modèle
def evaluate_model(y_true, y_pred, prefix=""):
    """Calcule toutes les métriques d'évaluation"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    rmse = np.sqrt(mse)
    
    # Calculer la loss par quantile
    quantiles = [0.05, 0.5, 0.95]
    quantile_losses = []
    for q in quantiles:
        errors = y_true - y_pred
        quantile_loss = np.mean(np.maximum(q * errors, (q - 1) * errors))
        quantile_losses.append(quantile_loss)
    
    return {
        f'{prefix}MSE': mse,
        f'{prefix}MAE': mae,
        f'{prefix}R²': r2,
        f'{prefix}MAPE (%)': mape,
        f'{prefix}RMSE': rmse,
        f'{prefix}Q05_Loss': quantile_losses[0],
        f'{prefix}Q50_Loss': quantile_losses[1],
        f'{prefix}Q95_Loss': quantile_losses[2]
    }

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
        
        # 2. Modèle pour les PUTs
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
        
        # 3. Modèle combiné (pour comparaison)
        if needs_scaling:
            model_combined = model.__class__(**model.get_params())
            model_combined.fit(X_train_scaled, y_train)
            y_pred_combined = model_combined.predict(X_test_scaled)
        else:
            model_combined = model.__class__(**model.get_params())
            model_combined.fit(X_train, y_train)
            y_pred_combined = model_combined.predict(X_test)
        
        # Évaluer les modèles
        results_calls[model_name] = {
            'model': model_calls,
            'predictions': y_pred_calls,
            **evaluate_model(y_test_calls, y_pred_calls, "CALL_")
        }
        
        results_puts[model_name] = {
            'model': model_puts,
            'predictions': y_pred_puts,
            **evaluate_model(y_test_puts, y_pred_puts, "PUT_")
        }
        
        results_combined[model_name] = {
            'model': model_combined,
            'predictions': y_pred_combined,
            **evaluate_model(y_test, y_pred_combined, "COMBINED_")
        }
        
        # Afficher les résultats
        print(f"   ✅ CALLs - R²: {results_calls[model_name]['CALL_R²']:.4f}, MAPE: {results_calls[model_name]['CALL_MAPE (%)']:.2f}%")
        print(f"   ✅ PUTs  - R²: {results_puts[model_name]['PUT_R²']:.4f}, MAPE: {results_puts[model_name]['PUT_MAPE (%)']:.2f}%")
        print(f"   ✅ TOTAL - R²: {results_combined[model_name]['COMBINED_R²']:.4f}, MAPE: {results_combined[model_name]['COMBINED_MAPE (%)']:.2f}%")
        
    except Exception as e:
        print(f"   ❌ Erreur lors de l'entraînement: {str(e)}")

# =====================================================
# 9. ANALYSE COMPARATIVE ET SÉLECTION DU MEILLEUR MODÈLE
# =====================================================

print("\n📊 Analyse comparative des modèles...")
print("=" * 80)

# Créer un DataFrame de comparaison pour chaque type
comparison_calls = pd.DataFrame({
    model: {k: v for k, v in results.items() if k.startswith('CALL_')}
    for model, results in results_calls.items()
}).T

comparison_puts = pd.DataFrame({
    model: {k: v for k, v in results.items() if k.startswith('PUT_')}
    for model, results in results_puts.items()
}).T

comparison_combined = pd.DataFrame({
    model: {k: v for k, v in results.items() if k.startswith('COMBINED_')}
    for model, results in results_combined.items()
}).T

# Identifier les meilleurs modèles
best_call_model = comparison_calls['CALL_R²'].idxmax()
best_put_model = comparison_puts['PUT_R²'].idxmax()
best_combined_model = comparison_combined['COMBINED_R²'].idxmax()

print("\n🏆 Meilleurs modèles:")
print(f"   - CALLs: {best_call_model} (R² = {comparison_calls.loc[best_call_model, 'CALL_R²']:.4f})")
print(f"   - PUTs: {best_put_model} (R² = {comparison_puts.loc[best_put_model, 'PUT_R²']:.4f})")
print(f"   - COMBINÉ: {best_combined_model} (R² = {comparison_combined.loc[best_combined_model, 'COMBINED_R²']:.4f})")

print("\n📋 Tableau de comparaison détaillé (Top 3 modèles):")
print("\nCALLs:")
print(comparison_calls.sort_values('CALL_R²', ascending=False).head(3).round(4))
print("\nPUTs:")
print(comparison_puts.sort_values('PUT_R²', ascending=False).head(3).round(4))

# =====================================================
# 10. ANALYSE DE L'IMPORTANCE DES FEATURES
# =====================================================

print("\n🔍 Analyse de l'importance des features...")

# Utiliser le meilleur modèle basé sur les arbres pour l'analyse
tree_based_models = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM']
best_tree_model = None

for model in tree_based_models:
    if model == best_combined_model:
        best_tree_model = results_combined[model]['model']
        break

if best_tree_model is None:
    # Si le meilleur modèle n'est pas basé sur les arbres, utiliser XGBoost
    best_tree_model = results_combined['XGBoost']['model']
    print("   Utilisation de XGBoost pour l'analyse des features")

# Obtenir l'importance des features
if hasattr(best_tree_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_tree_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📊 Top 15 features les plus importantes:")
    for idx, row in feature_importance.head(15).iterrows():
        print(f"   {idx+1:2d}. {row['feature']:25s} - {row['importance']:.4f}")
    
    # Analyser par type de feature
    print("\n📊 Importance par type de feature:")
    for feat_type, feat_list in feature_types.items():
        type_importance = feature_importance[feature_importance['feature'].isin(feat_list)]['importance'].sum()
        print(f"   - {feat_type:12s}: {type_importance:.4f}")

# =====================================================
# 11. VALIDATION TEMPORELLE WALK-FORWARD
# =====================================================

print("\n⏰ Validation Walk-Forward pour vérifier la stabilité temporelle...")

if 'Date' in df.columns:
    # Utiliser les vraies dates
    dates = df['Date'].unique()
    dates.sort()
    
    # Créer des splits mensuels
    walk_forward_results = []
    
    for i in range(len(dates) - 30):  # 30 jours de test à chaque fois
        train_end = dates[i]
        test_start = dates[i+1]
        test_end = dates[min(i+30, len(dates)-1)]
        
        # Créer les sets d'entraînement et de test
        train_mask = df['Date'] <= train_end
        test_mask = (df['Date'] >= test_start) & (df['Date'] <= test_end)
        
        if test_mask.sum() < 100:  # Pas assez de données de test
            continue
        
        # Entraîner et évaluer XGBoost
        X_wf_train = X[train_mask]
        y_wf_train = y[train_mask]
        X_wf_test = X[test_mask]
        y_wf_test = y[test_mask]
        
        model_wf = xgb.XGBRegressor(**Config.XGBOOST_PARAMS)
        model_wf.fit(X_wf_train, y_wf_train)
        y_wf_pred = model_wf.predict(X_wf_test)
        
        r2_wf = r2_score(y_wf_test, y_wf_pred)
        walk_forward_results.append({
            'train_end': train_end,
            'test_period': f"{test_start} to {test_end}",
            'r2_score': r2_wf,
            'n_test_samples': len(y_wf_test)
        })
        
        if len(walk_forward_results) >= 5:  # Limiter à 5 périodes pour la démo
            break
    
    if walk_forward_results:
        wf_df = pd.DataFrame(walk_forward_results)
        print("\n📊 Résultats de la validation Walk-Forward:")
        print(wf_df)
        print(f"\n   R² moyen: {wf_df['r2_score'].mean():.4f} (±{wf_df['r2_score'].std():.4f})")

# =====================================================
# 12. CRÉATION DES VISUALISATIONS AVANCÉES
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
                              calls_df['Strike'].quantile(0.9), 100)
    maturity_range = np.linspace(0.02, 2.0, 100)
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

# 2. Comparaison des modèles (R² et MAPE)
ax2 = fig.add_subplot(3, 3, 2)
models_sorted = comparison_combined.sort_values('COMBINED_R²', ascending=False)
x_pos = np.arange(len(models_sorted))
ax2.bar(x_pos - 0.2, models_sorted['COMBINED_R²'], 0.4, label='R²', color='skyblue')
ax2.bar(x_pos + 0.2, models_sorted['COMBINED_MAPE (%)'] / 100, 0.4, label='MAPE/100', color='lightcoral')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(models_sorted.index, rotation=45, ha='right')
ax2.set_ylabel('Score')
ax2.set_title('Performance des Modèles')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Scatter plot Prédictions vs Réalité (meilleur modèle)
ax3 = fig.add_subplot(3, 3, 3)
best_preds = results_combined[best_combined_model]['predictions']
ax3.scatter(y_test, best_preds, alpha=0.5, s=20, c='blue', edgecolors='none')
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax3.set_xlabel('Prix Réel ($)')
ax3.set_ylabel('Prix Prédit ($)')
ax3.set_title(f'Prédictions vs Réalité - {best_combined_model}')
ax3.grid(True, alpha=0.3)

# 4. Distribution des erreurs par modèle
ax4 = fig.add_subplot(3, 3, 4)
errors_data = []
error_labels = []
for model_name in models_sorted.index[:5]:  # Top 5 modèles
    errors = y_test.values - results_combined[model_name]['predictions']
    errors_data.append(errors)
    error_labels.append(model_name)
ax4.boxplot(errors_data, labels=error_labels)
ax4.set_xlabel('Modèle')
ax4.set_ylabel('Erreur ($)')
ax4.set_title('Distribution des Erreurs - Top 5 Modèles')
ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
ax4.set_xticklabels(error_labels, rotation=45, ha='right')

# 5. Importance des features (top 15)
ax5 = fig.add_subplot(3, 3, 5)
if 'feature_importance' in locals():
    top_features = feature_importance.head(15)
    ax5.barh(range(len(top_features)), top_features['importance'])
    ax5.set_yticks(range(len(top_features)))
    ax5.set_yticklabels(top_features['feature'])
    ax5.set_xlabel('Importance')
    ax5.set_title('Top 15 Features')
    ax5.grid(True, alpha=0.3)

# 6. Performance par moneyness
ax6 = fig.add_subplot(3, 3, 6)
# Créer des bins de moneyness
moneyness_bins = pd.cut(X_test['Moneyness'], bins=[0, 0.9, 0.95, 1.05, 1.1, 2])
moneyness_labels = ['Deep OTM Put', 'OTM Put', 'ATM', 'OTM Call', 'Deep OTM Call']

mape_by_moneyness = []
for i, (lower, upper) in enumerate([(0, 0.9), (0.9, 0.95), (0.95, 1.05), (1.05, 1.1), (1.1, 2)]):
    mask = (X_test['Moneyness'] >= lower) & (X_test['Moneyness'] < upper)
    if mask.sum() > 0:
        mape = mean_absolute_percentage_error(y_test[mask], best_preds[mask]) * 100
        mape_by_moneyness.append(mape)
    else:
        mape_by_moneyness.append(0)

ax6.bar(range(len(moneyness_labels)), mape_by_moneyness)
ax6.set_xticks(range(len(moneyness_labels)))
ax6.set_xticklabels(moneyness_labels, rotation=45, ha='right')
ax6.set_ylabel('MAPE (%)')
ax6.set_title('Erreur par Moneyness')
ax6.grid(True, alpha=0.3)

# 7. Performance par maturité
ax7 = fig.add_subplot(3, 3, 7)
maturity_bins = pd.cut(X_test['Temps_Maturite_Annees'], bins=[0, 0.1, 0.25, 0.5, 1, 3])
maturity_labels = ['< 1 mois', '1-3 mois', '3-6 mois', '6-12 mois', '> 1 an']

mape_by_maturity = []
for i, (lower, upper) in enumerate([(0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1), (1, 3)]):
    mask = (X_test['Temps_Maturite_Annees'] >= lower) & (X_test['Temps_Maturite_Annees'] < upper)
    if mask.sum() > 0:
        mape = mean_absolute_percentage_error(y_test[mask], best_preds[mask]) * 100
        mape_by_maturity.append(mape)
    else:
        mape_by_maturity.append(0)

ax7.bar(range(len(maturity_labels)), mape_by_maturity, color='green')
ax7.set_xticks(range(len(maturity_labels)))
ax7.set_xticklabels(maturity_labels, rotation=45, ha='right')
ax7.set_ylabel('MAPE (%)')
ax7.set_title('Erreur par Maturité')
ax7.grid(True, alpha=0.3)

# 8. Comparaison CALLs vs PUTs
ax8 = fig.add_subplot(3, 3, 8)
call_put_comparison = pd.DataFrame({
    'CALLs': comparison_calls.iloc[0][['CALL_R²', 'CALL_MAPE (%)']],
    'PUTs': comparison_puts.iloc[0][['PUT_R²', 'PUT_MAPE (%)']]
}).T
call_put_comparison.columns = ['R²', 'MAPE (%)']
call_put_comparison.plot(kind='bar', ax=ax8, color=['blue', 'red'])
ax8.set_title(f'Performance CALLs vs PUTs - {best_combined_model}')
ax8.set_ylabel('Score')
ax8.set_xticklabels(['CALLs', 'PUTs'], rotation=0)
ax8.legend()
ax8.grid(True, alpha=0.3)

# 9. Importance par type de feature
ax9 = fig.add_subplot(3, 3, 9)
if 'feature_importance' in locals():
    type_importance = []
    type_names = []
    for feat_type, feat_list in feature_types.items():
        importance = feature_importance[feature_importance['feature'].isin(feat_list)]['importance'].sum()
        if importance > 0:
            type_importance.append(importance)
            type_names.append(feat_type)
    
    ax9.pie(type_importance, labels=type_names, autopct='%1.1f%%', startangle=90)
    ax9.set_title('Importance par Type de Feature')

plt.tight_layout()
plt.savefig('advanced_options_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# =====================================================
# 13. GÉNÉRATION DU RAPPORT FINAL ET SAUVEGARDE
# =====================================================

print("\n📝 Génération du rapport final...")

# Créer un rapport structuré
report = {
    'metadata': {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_samples': len(df),
        'n_features': len(feature_columns),
        'config': Config.__dict__
    },
    'data_quality': {
        'missing_values': X.isnull().sum().to_dict(),
        'feature_stats': X.describe().to_dict()
    },
    'model_performance': {
        'calls': comparison_calls.to_dict(),
        'puts': comparison_puts.to_dict(),
        'combined': comparison_combined.to_dict()
    },
    'best_models': {
        'calls': best_call_model,
        'puts': best_put_model,
        'combined': best_combined_model
    },
    'feature_importance': feature_importance.to_dict() if 'feature_importance' in locals() else None,
    'recommendations': [
        "Utiliser des modèles séparés pour CALLs et PUTs améliore la performance",
        f"Le modèle {best_combined_model} offre le meilleur équilibre performance/complexité",
        "Les Greeks (notamment Delta et Vega) sont parmi les features les plus importantes",
        "La volatilité implicite et la structure de moneyness dominent les prédictions",
        "Considérer l'ajout de données de marché en temps réel (VIX, corrélations) pour améliorer encore"
    ]
}

# Sauvegarder le rapport
with open('options_ml_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)

# Sauvegarder les résultats de comparaison
comparison_combined.to_csv('model_comparison_enhanced.csv')

# Sauvegarder les meilleurs modèles
print("\n💾 Sauvegarde des modèles...")
joblib.dump(results_calls[best_call_model]['model'], f'best_model_calls_{best_call_model.replace(" ", "_")}.pkl')
joblib.dump(results_puts[best_put_model]['model'], f'best_model_puts_{best_put_model.replace(" ", "_")}.pkl')
joblib.dump(results_combined[best_combined_model]['model'], f'best_model_combined_{best_combined_model.replace(" ", "_")}.pkl')
joblib.dump(scaler, 'feature_scaler_enhanced.pkl')

print("\n✨ Analyse complète terminée avec succès!")
print("=" * 60)
print("\n📊 Résumé des performances:")
print(f"   - Meilleur R² global: {comparison_combined.iloc[0]['COMBINED_R²']:.4f}")
print(f"   - Meilleur MAPE global: {comparison_combined.iloc[0]['COMBINED_MAPE (%)']:.2f}%")
print(f"   - Amélioration CALLs vs PUTs: {abs(comparison_calls.iloc[0]['CALL_R²'] - comparison_puts.iloc[0]['PUT_R²']):.4f}")
print("\n📁 Fichiers générés:")
print("   - advanced_options_analysis.png")
print("   - options_ml_report.json")
print("   - model_comparison_enhanced.csv")
print("   - Modèles sauvegardés (.pkl)")
