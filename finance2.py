"""
Script pédagogique pour récupérer et analyser les données d'options AAPL
Auteur: Assistant Claude
Date: Mai 2025
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# 1. CONFIGURATION INITIALE
# =====================================================

print("📊 Démarrage de l'analyse des options AAPL")
print("=" * 50)

# Création de l'objet ticker pour Apple
ticker = yf.Ticker("AAPL")

# =====================================================
# 2. RÉCUPÉRATION DES DONNÉES DU SOUS-JACENT
# =====================================================

print("\n📈 Récupération des données historiques du sous-jacent AAPL...")

# Obtenir les données sur 7 ans pour le sous-jacent
end_date = datetime.now()
start_date = end_date - timedelta(days=7*365)

# Données historiques du cours de l'action
stock_history = ticker.history(start=start_date, end=end_date)

print(f"✅ Données récupérées du {start_date.strftime('%Y-%m-%d')} au {end_date.strftime('%Y-%m-%d')}")
print(f"   Nombre de jours de trading: {len(stock_history)}")

# =====================================================
# 3. RÉCUPÉRATION DES DATES D'EXPIRATION DISPONIBLES
# =====================================================

print("\n📅 Récupération des dates d'expiration disponibles...")

try:
    # Obtenir toutes les dates d'expiration disponibles
    expiration_dates = ticker.options
    print(f"✅ {len(expiration_dates)} dates d'expiration trouvées")
    print(f"   Première date: {expiration_dates[0]}")
    print(f"   Dernière date: {expiration_dates[-1]}")
except Exception as e:
    print(f"❌ Erreur lors de la récupération des dates: {e}")
    expiration_dates = []

# Fonction utilitaire pour analyser la distribution des maturités disponibles
def analyze_maturity_distribution(exp_dates):
    """Analyse la distribution des dates d'expiration disponibles"""
    current_date = datetime.now()
    maturity_days = []
    
    for exp_date in exp_dates:
        exp_datetime = pd.to_datetime(exp_date)
        days_to_expiry = (exp_datetime - current_date).days
        maturity_days.append(days_to_expiry)
    
    maturity_df = pd.DataFrame({
        'Date_Expiration': exp_dates,
        'Jours_Jusqu_Expiration': maturity_days
    })
    
    print("\n📊 Distribution des maturités disponibles:")
    print(f"   - Moins d'1 semaine: {len(maturity_df[maturity_df['Jours_Jusqu_Expiration'] < 7])}")
    print(f"   - 1 semaine à 1 mois: {len(maturity_df[(maturity_df['Jours_Jusqu_Expiration'] >= 7) & (maturity_df['Jours_Jusqu_Expiration'] <= 30)])}")
    print(f"   - 1 à 3 mois: {len(maturity_df[(maturity_df['Jours_Jusqu_Expiration'] > 30) & (maturity_df['Jours_Jusqu_Expiration'] <= 90)])}")
    print(f"   - 3 mois à 1 an: {len(maturity_df[(maturity_df['Jours_Jusqu_Expiration'] > 90) & (maturity_df['Jours_Jusqu_Expiration'] <= 365)])}")
    print(f"   - 1 à 3 ans: {len(maturity_df[(maturity_df['Jours_Jusqu_Expiration'] > 365) & (maturity_df['Jours_Jusqu_Expiration'] <= 1095)])}")
    print(f"   - Plus de 3 ans: {len(maturity_df[maturity_df['Jours_Jusqu_Expiration'] > 1095])}")
    
    return maturity_df

# Analyser la distribution si des dates sont disponibles
if expiration_dates:
    maturity_analysis = analyze_maturity_distribution(expiration_dates)

# =====================================================
# 4. FONCTION POUR CALCULER LA VOLATILITÉ HISTORIQUE
# =====================================================

def calculate_historical_volatility(prices, period=30):
    """
    Calcule la volatilité historique sur une période donnée
    
    Paramètres:
    - prices: Series pandas des prix de clôture
    - period: Nombre de jours pour le calcul (défaut: 30)
    
    Retourne:
    - Volatilité annualisée en pourcentage
    """
    # Calcul des rendements logarithmiques
    returns = np.log(prices / prices.shift(1))
    
    # Calcul de l'écart-type des rendements
    volatility = returns.rolling(window=period).std()
    
    # Annualisation (252 jours de trading par an)
    annualized_volatility = volatility * np.sqrt(252) * 100
    
    return annualized_volatility

# Calcul de la volatilité historique sur 30 jours
stock_history['Volatility_30d'] = calculate_historical_volatility(stock_history['Close'])

# =====================================================
# 5. RÉCUPÉRATION DES DONNÉES D'OPTIONS
# =====================================================

print("\n🎯 Récupération des données d'options...")

# Paramètres de filtrage par maturité
MIN_DAYS_TO_EXPIRY = 7  # 1 semaine minimum
MAX_DAYS_TO_EXPIRY = 365 * 3  # 3 ans maximum

print(f"   ⏱️  Filtre de maturité: {MIN_DAYS_TO_EXPIRY} jours à {MAX_DAYS_TO_EXPIRY} jours")

# Dictionnaire pour stocker toutes les données d'options
all_options_data = []

# Filtrer les dates d'expiration selon les critères de maturité
filtered_expirations = []
current_date = datetime.now()

for exp_date in expiration_dates:
    exp_datetime = pd.to_datetime(exp_date)
    days_to_expiry = (exp_datetime - current_date).days
    
    if MIN_DAYS_TO_EXPIRY <= days_to_expiry <= MAX_DAYS_TO_EXPIRY:
        filtered_expirations.append(exp_date)

print(f"   📅 {len(filtered_expirations)} dates d'expiration correspondent aux critères")
print(f"      (sur {len(expiration_dates)} dates totales disponibles)")

# Limiter à quelques dates d'expiration pour l'exemple
# (vous pouvez augmenter cette limite selon vos besoins)
sample_expirations = filtered_expirations[:10] if len(filtered_expirations) > 10 else filtered_expirations

for exp_date in sample_expirations:
    print(f"\n  📍 Traitement de l'expiration: {exp_date}")
    
    try:
        # Récupérer la chaîne d'options pour cette date
        opt_chain = ticker.option_chain(exp_date)
        
        # Traiter les calls
        calls = opt_chain.calls
        calls['optionType'] = 'CALL'
        calls['expirationDate'] = exp_date
        
        # Traiter les puts
        puts = opt_chain.puts
        puts['optionType'] = 'PUT'
        puts['expirationDate'] = exp_date
        
        # Combiner calls et puts
        options_df = pd.concat([calls, puts], ignore_index=True)
        
        # Ajouter le prix actuel du sous-jacent
        current_price = stock_history['Close'].iloc[-1]
        options_df['underlyingPrice'] = current_price
        
        # Calculer le temps jusqu'à maturité en années
        exp_datetime = pd.to_datetime(exp_date)
        days_to_expiry = (exp_datetime - datetime.now()).days
        options_df['timeToMaturity'] = days_to_expiry / 365.0
        
        # Ajouter la volatilité actuelle
        options_df['historicalVolatility'] = stock_history['Volatility_30d'].iloc[-1]
        
        # Stocker les données
        all_options_data.append(options_df)
        
        print(f"     ✅ {len(options_df)} options récupérées")
        
    except Exception as e:
        print(f"     ❌ Erreur: {e}")

# =====================================================
# 6. CONSOLIDATION ET NETTOYAGE DES DONNÉES
# =====================================================

print("\n🔧 Consolidation des données...")

if all_options_data:
    # Combiner toutes les données d'options
    final_options_df = pd.concat(all_options_data, ignore_index=True)
    
    # Sélectionner et renommer les colonnes importantes
    columns_mapping = {
        'strike': 'Strike',
        'lastPrice': 'Prix_Option',
        'impliedVolatility': 'Volatilite_Implicite',
        'volume': 'Volume',
        'openInterest': 'Open_Interest',
        'bid': 'Bid',
        'ask': 'Ask',
        'optionType': 'Type_Option',
        'expirationDate': 'Date_Expiration',
        'underlyingPrice': 'Prix_Sous_Jacent',
        'timeToMaturity': 'Temps_Maturite_Annees',
        'historicalVolatility': 'Volatilite_Historique'
    }
    
    final_options_df = final_options_df[list(columns_mapping.keys())].rename(columns=columns_mapping)
    
    # Convertir la volatilité implicite en pourcentage
    final_options_df['Volatilite_Implicite'] = final_options_df['Volatilite_Implicite'] * 100
    
    # Calculer la moneyness (rapport strike/sous-jacent)
    final_options_df['Moneyness'] = final_options_df['Strike'] / final_options_df['Prix_Sous_Jacent']
    
    # Filtrer les options avec des données valides
    final_options_df = final_options_df[
        (final_options_df['Prix_Option'] > 0) & 
        (final_options_df['Volatilite_Implicite'] > 0)
    ]
    
    # Appliquer le filtre de maturité sur le dataframe final
    # Convertir le temps de maturité en jours pour le filtrage
    final_options_df['Jours_Jusqu_Maturite'] = final_options_df['Temps_Maturite_Annees'] * 365
    
    # Filtrer selon les critères de maturité
    options_before_filter = len(final_options_df)
    final_options_df = final_options_df[
        (final_options_df['Jours_Jusqu_Maturite'] >= MIN_DAYS_TO_EXPIRY) &
        (final_options_df['Jours_Jusqu_Maturite'] <= MAX_DAYS_TO_EXPIRY)
    ]
    options_after_filter = len(final_options_df)
    
    print(f"✅ {options_after_filter} options consolidées avec succès")
    print(f"   📊 Filtrage maturité: {options_before_filter} → {options_after_filter} options")
    print(f"   ⏱️  Plage de maturité: {final_options_df['Jours_Jusqu_Maturite'].min():.0f} - {final_options_df['Jours_Jusqu_Maturite'].max():.0f} jours")
    
    # =====================================================
    # 7. ANALYSE ET STATISTIQUES
    # =====================================================
    
    print("\n📊 Statistiques des options récupérées:")
    print("=" * 50)
    
    # Statistiques sur les maturités
    print("\n⏱️  DISTRIBUTION DES MATURITÉS:")
    print(f"  - Maturité minimale: {final_options_df['Jours_Jusqu_Maturite'].min():.0f} jours")
    print(f"  - Maturité maximale: {final_options_df['Jours_Jusqu_Maturite'].max():.0f} jours")
    print(f"  - Maturité moyenne: {final_options_df['Jours_Jusqu_Maturite'].mean():.0f} jours")
    
    # Catégoriser les maturités
    def categorize_maturity(days):
        if days <= 30:
            return 'Court terme (≤ 1 mois)'
        elif days <= 90:
            return 'Moyen terme (1-3 mois)'
        elif days <= 365:
            return 'Long terme (3-12 mois)'
        else:
            return 'Très long terme (> 1 an)'
    
    final_options_df['Categorie_Maturite'] = final_options_df['Jours_Jusqu_Maturite'].apply(categorize_maturity)
    
    print("\n  Distribution par catégorie de maturité:")
    maturity_dist = final_options_df['Categorie_Maturite'].value_counts()
    for cat, count in maturity_dist.items():
        print(f"    - {cat}: {count} options ({count/len(final_options_df)*100:.1f}%)")
    
    # Statistiques par type d'option
    print("\n📈 STATISTIQUES PAR TYPE D'OPTION:")
    for opt_type in ['CALL', 'PUT']:
        subset = final_options_df[final_options_df['Type_Option'] == opt_type]
        if len(subset) > 0:
            print(f"\n  {opt_type}S:")
            print(f"    - Nombre: {len(subset)}")
            print(f"    - Prix moyen: ${subset['Prix_Option'].mean():.2f}")
            print(f"    - Volatilité implicite moyenne: {subset['Volatilite_Implicite'].mean():.2f}%")
            print(f"    - Strike min/max: ${subset['Strike'].min():.2f} - ${subset['Strike'].max():.2f}")
            print(f"    - Maturité moyenne: {subset['Jours_Jusqu_Maturite'].mean():.0f} jours")
    
    # =====================================================
    # 8. CRÉATION D'UN ÉCHANTILLON DIVERSIFIÉ
    # =====================================================
    
    print("\n🎲 Création d'un échantillon diversifié...")
    
    # Créer des catégories de moneyness
    def categorize_moneyness(moneyness):
        if moneyness < 0.95:
            return 'ITM'  # In The Money
        elif moneyness <= 1.05:
            return 'ATM'  # At The Money
        else:
            return 'OTM'  # Out of The Money
    
    final_options_df['Categorie_Moneyness'] = final_options_df['Moneyness'].apply(categorize_moneyness)
    
    # Échantillonner des options de chaque catégorie (moneyness x maturité x type)
    sample_size = 3  # Nombre d'options par combinaison
    diversified_sample = pd.DataFrame()
    
    # Créer toutes les combinaisons possibles
    moneyness_categories = ['ITM', 'ATM', 'OTM']
    maturity_categories = final_options_df['Categorie_Maturite'].unique()
    option_types = ['CALL', 'PUT']
    
    for mat_cat in maturity_categories:
        for money_cat in moneyness_categories:
            for opt_type in option_types:
                subset = final_options_df[
                    (final_options_df['Categorie_Maturite'] == mat_cat) &
                    (final_options_df['Categorie_Moneyness'] == money_cat) & 
                    (final_options_df['Type_Option'] == opt_type)
                ]
                if len(subset) > 0:
                    sample = subset.sample(n=min(sample_size, len(subset)))
                    diversified_sample = pd.concat([diversified_sample, sample])
    
    print(f"✅ Échantillon diversifié créé avec {len(diversified_sample)} options")
    print(f"   Répartition par maturité:")
    for cat, count in diversified_sample['Categorie_Maturite'].value_counts().items():
        print(f"   - {cat}: {count} options")
    
    # =====================================================
    # 9. AFFICHAGE DES RÉSULTATS
    # =====================================================
    
    print("\n📋 Échantillon des données récupérées:")
    print("=" * 100)
    
    # Colonnes à afficher
    display_columns = [
        'Type_Option', 'Strike', 'Prix_Option', 'Prix_Sous_Jacent',
        'Volatilite_Implicite', 'Jours_Jusqu_Maturite', 'Categorie_Maturite', 'Categorie_Moneyness'
    ]
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 20)
    
    print(diversified_sample[display_columns].head(10).to_string(index=False))
    
    # =====================================================
    # 10. SAUVEGARDE DES DONNÉES
    # =====================================================
    
    print("\n💾 Sauvegarde des données...")
    
    # Sauvegarder toutes les données
    final_options_df.to_csv('aapl_options_data_complete.csv', index=False)
    print("✅ Données complètes sauvegardées dans 'aapl_options_data_complete.csv'")
    
    # Sauvegarder l'échantillon diversifié
    diversified_sample.to_csv('aapl_options_sample.csv', index=False)
    print("✅ Échantillon diversifié sauvegardé dans 'aapl_options_sample.csv'")
    
    # =====================================================
    # 11. NOTE SUR LE TAUX SANS RISQUE
    # =====================================================
    
    print("\n💡 Note importante:")
    print("   Le taux sans risque n'est pas directement disponible via yfinance.")
    print("   Pour une analyse complète, vous devriez:")
    print("   - Utiliser le taux des bons du Trésor US (ex: ^TNX pour le 10 ans)")
    print("   - Ou utiliser une valeur fixe (ex: 5% actuellement)")
    
    # Exemple d'ajout du taux sans risque
    risk_free_rate = 0.05  # 5% comme exemple
    final_options_df['Taux_Sans_Risque'] = risk_free_rate
    
else:
    print("❌ Aucune donnée d'options n'a pu être récupérée")

print("\n✨ Analyse terminée!")
print("=" * 50)

# =====================================================
# 12. FONCTION BONUS: DONNÉES HISTORIQUES SIMULÉES
# =====================================================

def simulate_historical_options_data(stock_history, num_years=7):
    """
    Simule des données d'options historiques basées sur le mouvement du sous-jacent
    Cette fonction est utile car yfinance ne fournit pas d'historique d'options
    
    Note: Ceci est une SIMULATION à des fins pédagogiques
    """
    print("\n🎯 Simulation de données historiques d'options (à titre pédagogique)...")
    
    historical_options = []
    
    # Générer des données mensuelles sur la période
    for i in range(0, len(stock_history), 21):  # ~21 jours de trading par mois
        date = stock_history.index[i]
        spot_price = stock_history['Close'].iloc[i]
        volatility = stock_history['Volatility_30d'].iloc[i] if not pd.isna(stock_history['Volatility_30d'].iloc[i]) else 25
        
        # Générer des strikes autour du prix spot
        strikes = np.linspace(spot_price * 0.8, spot_price * 1.2, 9)
        
        for strike in strikes:
            for option_type in ['CALL', 'PUT']:
                # Simulation simplifiée du prix (Black-Scholes simplifié)
                moneyness = strike / spot_price
                time_to_expiry = 0.25  # 3 mois
                
                # Prix approximatif basé sur la moneyness
                if option_type == 'CALL':
                    intrinsic_value = max(0, spot_price - strike)
                else:
                    intrinsic_value = max(0, strike - spot_price)
                
                time_value = spot_price * 0.1 * np.sqrt(time_to_expiry) * volatility / 100
                option_price = intrinsic_value + time_value * np.random.uniform(0.8, 1.2)
                
                historical_options.append({
                    'Date': date,
                    'Type_Option': option_type,
                    'Strike': strike,
                    'Prix_Option': option_price,
                    'Prix_Sous_Jacent': spot_price,
                    'Volatilite_Historique': volatility,
                    'Temps_Maturite': time_to_expiry,
                    'Moneyness': moneyness
                })
    
    historical_df = pd.DataFrame(historical_options)
    print(f"✅ {len(historical_df)} points de données historiques simulés")
    
    return historical_df

# Exécuter la simulation si vous voulez des données historiques
# historical_simulated = simulate_historical_options_data(stock_history)
# historical_simulated.to_csv('aapl_options_historical_simulated.csv', index=False)
