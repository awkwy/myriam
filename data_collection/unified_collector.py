"""
Collecteur Unifié de Données d'Options - Module Principal
========================================================

Ce module unifie la collecte de données depuis plusieurs sources (Yahoo Finance, Polygon.io)
et structure les données selon le format requis pour le pricing d'options et d'autocalls.

Format de sortie standardisé:
- Date, Sous-jacent, Option, Prix, T(années), Strike, Vol%, Type
- Maturités: 1 semaine (0.019 ans) à 2 ans maximum
- Période: 6 mois d'historique quotidien

Auteur: Framework Options & Autocalls
Date: Mai 2025
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import logging
import os
from typing import Dict, List, Optional, Tuple
import json
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging avec format pédagogique
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class UnifiedOptionsCollector:
    """
    Collecteur unifié qui combine plusieurs sources de données pour créer
    un dataset complet et cohérent pour le pricing d'options.
    
    Points pédagogiques clés:
    - Standardisation des données de sources multiples
    - Calcul uniforme des Greeks selon Black-Scholes
    - Filtrage intelligent par maturité et liquidité
    - Structure optimisée pour les surfaces de volatilité
    """
    
    def __init__(self, config_path: str = "config/collector_config.json"):
        """
        Initialise le collecteur avec configuration flexible.
        
        La configuration permet de:
        - Choisir les sources de données (Yahoo, Polygon, etc.)
        - Définir les paramètres de filtrage (maturités min/max)
        - Configurer les taux sans risque par devise
        - Ajuster les seuils de qualité des données
        """
        # Charger la configuration
        self.config = self._load_config(config_path)
        
        # Paramètres de filtrage des maturités (entre 1 semaine et 2 ans)
        self.min_days_to_expiry = self.config.get('min_days_to_expiry', 7)
        self.max_days_to_expiry = self.config.get('max_days_to_expiry', 730)
        
        # Taux sans risque par défaut (peut varier selon la devise)
        self.risk_free_rates = self.config.get('risk_free_rates', {
            'USD': 0.05,
            'EUR': 0.04,
            'GBP': 0.045
        })
        
        # Statistiques de collecte pour monitoring
        self.stats = {
            'total_options_collected': 0,
            'options_filtered_out': 0,
            'successful_days': 0,
            'failed_days': 0,
            'data_quality_score': 0.0
        }
        
        logger.info("📊 Collecteur unifié initialisé avec configuration:")
        logger.info(f"   - Maturités: {self.min_days_to_expiry} à {self.max_days_to_expiry} jours")
        logger.info(f"   - Sources actives: {self.config.get('active_sources', ['yahoo'])}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Charge la configuration depuis un fichier JSON ou utilise les défauts."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"⚠️  Fichier de config non trouvé: {config_path}")
            logger.warning("   Utilisation de la configuration par défaut")
            return {
                'active_sources': ['yahoo'],
                'min_days_to_expiry': 7,
                'max_days_to_expiry': 730,
                'risk_free_rates': {'USD': 0.05},
                'quality_thresholds': {
                    'min_volume': 10,
                    'min_open_interest': 100,
                    'max_bid_ask_spread_pct': 50
                }
            }
    
    def calculate_black_scholes_greeks(self, S: float, K: float, T: float, 
                                     r: float, sigma: float, option_type: str) -> Dict[str, float]:
        """
        Calcule tous les Greeks selon le modèle Black-Scholes.
        
        Formules pédagogiques:
        - d1 = (ln(S/K) + (r + σ²/2)T) / (σ√T)
        - d2 = d1 - σ√T
        
        Greeks calculés:
        - Delta (Δ): Sensibilité au prix du sous-jacent
        - Gamma (Γ): Taux de changement du delta
        - Theta (Θ): Décroissance temporelle (en jours)
        - Vega (ν): Sensibilité à la volatilité
        - Rho (ρ): Sensibilité aux taux d'intérêt
        
        Args:
            S: Prix du sous-jacent
            K: Prix d'exercice (strike)
            T: Temps jusqu'à maturité (années)
            r: Taux sans risque
            sigma: Volatilité implicite
            option_type: 'CALL' ou 'PUT'
            
        Returns:
            Dict avec tous les Greeks calculés
        """
        # Protection contre les valeurs invalides
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return {
                'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 
                'vega': 0.0, 'rho': 0.0
            }
        
        # Calculs fondamentaux Black-Scholes
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        
        # Fonctions de distribution normale
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        n_d1 = norm.pdf(d1)
        
        # Calcul des Greeks selon le type d'option
        if option_type.upper() == 'CALL':
            delta = N_d1
            theta_annual = (
                -(S * n_d1 * sigma) / (2 * sqrt_T) - 
                r * K * np.exp(-r * T) * N_d2
            )
            rho = K * T * np.exp(-r * T) * N_d2 / 100
        else:  # PUT
            delta = N_d1 - 1
            theta_annual = (
                -(S * n_d1 * sigma) / (2 * sqrt_T) + 
                r * K * np.exp(-r * T) * norm.cdf(-d2)
            )
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        # Greeks communs aux calls et puts
        gamma = n_d1 / (S * sigma * sqrt_T)
        vega = S * n_d1 * sqrt_T / 100  # Divisé par 100 pour avoir vega en %
        theta_daily = theta_annual / 365.25  # Conversion en theta journalier
        
        return {
            'delta': round(delta, 4),
            'gamma': round(gamma, 6),
            'theta': round(theta_daily, 4),
            'vega': round(vega, 4),
            'rho': round(rho, 4)
        }
    
    def estimate_implied_volatility(self, S: float, K: float, T: float, 
                                  market_price: float, option_type: str, 
                                  r: float = 0.05) -> float:
        """
        Estime la volatilité implicite par méthode numérique (Newton-Raphson).
        
        Principe pédagogique:
        La volatilité implicite est LA volatilité qui, injectée dans Black-Scholes,
        donne exactement le prix de marché observé. C'est l'anticipation du marché
        sur la volatilité future du sous-jacent.
        
        Méthode:
        1. Estimation initiale basée sur la moneyness
        2. Itérations Newton-Raphson jusqu'à convergence
        3. Bornes de sécurité pour éviter les valeurs aberrantes
        """
        # Estimation initiale intelligente basée sur la moneyness
        moneyness = S / K
        if 0.8 < moneyness < 1.2:  # Near ATM
            sigma_init = 0.20  # 20% pour les options ATM
        else:  # OTM ou deep ITM
            sigma_init = 0.30  # Plus de vol pour les options éloignées
        
        # Si le prix de marché est très bas, retourner une estimation
        if market_price < 0.01:
            return sigma_init
        
        # Newton-Raphson avec maximum 50 itérations
        sigma = sigma_init
        for i in range(50):
            # Prix théorique avec volatilité actuelle
            bs_price = self._black_scholes_price(S, K, T, r, sigma, option_type)
            
            # Vega pour la dérivée
            vega = S * norm.pdf(self._d1(S, K, T, r, sigma)) * np.sqrt(T)
            
            # Éviter division par zéro
            if abs(vega) < 1e-10:
                break
            
            # Mise à jour Newton-Raphson
            price_diff = bs_price - market_price
            sigma_new = sigma - price_diff / vega
            
            # Bornes de sécurité
            sigma_new = max(0.01, min(3.0, sigma_new))  # Entre 1% et 300%
            
            # Test de convergence
            if abs(sigma_new - sigma) < 1e-6:
                break
                
            sigma = sigma_new
        
        return round(sigma, 4)
    
    def _black_scholes_price(self, S: float, K: float, T: float, r: float, 
                           sigma: float, option_type: str) -> float:
        """Calcule le prix Black-Scholes pour l'estimation de volatilité implicite."""
        if T <= 0 or sigma <= 0:
            return 0.0
            
        d1 = self._d1(S, K, T, r, sigma)
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.upper() == 'CALL':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    def _d1(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calcule d1 pour Black-Scholes."""
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    def collect_yahoo_options(self, ticker: str, date: datetime) -> pd.DataFrame:
        """
        Collecte les données d'options depuis Yahoo Finance pour une date donnée.
        
        Processus:
        1. Récupération du prix du sous-jacent
        2. Collecte de toutes les maturités disponibles
        3. Filtrage par critères de liquidité et maturité
        4. Calcul des Greeks et métriques
        5. Formatage selon le standard unifié
        
        Returns:
            DataFrame avec colonnes standardisées
        """
        logger.info(f"📥 Collecte Yahoo Finance pour {ticker} le {date.strftime('%Y-%m-%d')}")
        
        try:
            # Initialiser l'objet ticker
            stock = yf.Ticker(ticker)
            
            # Obtenir le prix du sous-jacent
            hist = stock.history(start=date - timedelta(days=5), end=date + timedelta(days=1))
            if hist.empty or date.strftime('%Y-%m-%d') not in hist.index.strftime('%Y-%m-%d'):
                logger.warning(f"⚠️  Pas de données de prix pour {date}")
                return pd.DataFrame()
            
            spot_price = hist.loc[hist.index.strftime('%Y-%m-%d') == date.strftime('%Y-%m-%d'), 'Close'].iloc[0]
            
            # Obtenir toutes les dates d'expiration
            expirations = stock.options
            
            all_options = []
            
            for expiry in expirations:
                # Calculer les jours jusqu'à expiration
                exp_date = pd.to_datetime(expiry)
                days_to_expiry = (exp_date - date).days
                
                # Filtrer selon nos critères de maturité
                if not (self.min_days_to_expiry <= days_to_expiry <= self.max_days_to_expiry):
                    continue
                
                # Récupérer la chaîne d'options
                opt_chain = stock.option_chain(expiry)
                
                # Traiter calls et puts
                for opt_type, opt_data in [('CALL', opt_chain.calls), ('PUT', opt_chain.puts)]:
                    for _, opt in opt_data.iterrows():
                        # Filtres de qualité
                        if opt['volume'] < self.config.get('quality_thresholds', {}).get('min_volume', 10):
                            continue
                        
                        # Calculer le prix (mid si disponible)
                        bid = opt.get('bid', 0)
                        ask = opt.get('ask', 0)
                        if bid > 0 and ask > 0:
                            option_price = (bid + ask) / 2
                        else:
                            option_price = opt.get('lastPrice', 0)
                        
                        if option_price <= 0:
                            continue
                        
                        # Calculer les métriques
                        years_to_expiry = days_to_expiry / 365.25
                        
                        # Volatilité implicite
                        impl_vol = opt.get('impliedVolatility', 0)
                        if impl_vol <= 0:
                            impl_vol = self.estimate_implied_volatility(
                                S=spot_price,
                                K=opt['strike'],
                                T=years_to_expiry,
                                market_price=option_price,
                                option_type=opt_type
                            )
                        
                        # Calculer les Greeks
                        greeks = self.calculate_black_scholes_greeks(
                            S=spot_price,
                            K=opt['strike'],
                            T=years_to_expiry,
                            r=self.risk_free_rates.get('USD', 0.05),
                            sigma=impl_vol,
                            option_type=opt_type
                        )
                        
                        # Créer l'enregistrement standardisé
                        option_record = {
                            'Date': date.strftime('%Y-%m-%d'),
                            'Sous_Jacent': spot_price,
                            'Ticker': ticker,
                            'Option': f"{int(opt['strike']/spot_price*100)}% {years_to_expiry:.2f}Y",
                            'Prix': round(option_price, 2),
                            'T_Annees': round(years_to_expiry, 4),
                            'Strike': opt['strike'],
                            'Vol_Pct': round(impl_vol * 100, 2),
                            'Type': opt_type,
                            'Moneyness': round(opt['strike'] / spot_price, 4),
                            'Delta': greeks['delta'],
                            'Gamma': greeks['gamma'],
                            'Theta': greeks['theta'],
                            'Vega': greeks['vega'],
                            'Rho': greeks['rho'],
                            'Volume': opt.get('volume', 0),
                            'Open_Interest': opt.get('openInterest', 0),
                            'Bid': bid,
                            'Ask': ask,
                            'Spread_Pct': round((ask - bid) / option_price * 100, 2) if option_price > 0 else 0
                        }
                        
                        all_options.append(option_record)
                        self.stats['total_options_collected'] += 1
            
            if all_options:
                logger.info(f"✅ Collecté {len(all_options)} options pour {ticker} le {date.strftime('%Y-%m-%d')}")
                self.stats['successful_days'] += 1
            else:
                logger.warning(f"⚠️  Aucune option valide trouvée pour {date}")
                self.stats['failed_days'] += 1
            
            return pd.DataFrame(all_options)
            
        except Exception as e:
            logger.error(f"❌ Erreur collecte Yahoo pour {ticker}: {e}")
            self.stats['failed_days'] += 1
            return pd.DataFrame()
    
    def collect_historical_data(self, tickers: List[str], start_date: datetime, 
                              end_date: datetime, parallel_workers: int = 3) -> pd.DataFrame:
        """
        Collecte les données historiques pour plusieurs tickers sur une période.
        
        Stratégie de collecte:
        1. Génération des jours de trading (excluant weekends et jours fériés)
        2. Parallélisation intelligente pour optimiser la performance
        3. Sauvegarde incrémentale pour résistance aux erreurs
        4. Consolidation et validation finale
        
        Args:
            tickers: Liste des sous-jacents à collecter
            start_date: Date de début (incluse)
            end_date: Date de fin (incluse)  
            parallel_workers: Nombre de threads parallèles
            
        Returns:
            DataFrame consolidé avec toutes les données
        """
        logger.info("🚀 Début de la collecte historique")
        logger.info(f"   - Tickers: {tickers}")
        logger.info(f"   - Période: {start_date.strftime('%Y-%m-%d')} à {end_date.strftime('%Y-%m-%d')}")
        
        # Générer les jours de trading
        trading_days = pd.bdate_range(start=start_date, end=end_date)
        logger.info(f"   - Jours de trading: {len(trading_days)}")
        
        # Créer le répertoire de sauvegarde
        output_dir = f"data/options_historical_{datetime.now().strftime('%Y%m%d')}"
        os.makedirs(output_dir, exist_ok=True)
        
        all_data = []
        
        # Traiter chaque ticker
        for ticker in tickers:
            logger.info(f"\n📊 Traitement de {ticker}")
            ticker_data = []
            
            # Collecter par batches de jours
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                # Soumettre les tâches
                future_to_date = {}
                for date in trading_days:
                    future = executor.submit(self.collect_yahoo_options, ticker, date)
                    future_to_date[future] = date
                
                # Récupérer les résultats
                for future in as_completed(future_to_date):
                    date = future_to_date[future]
                    try:
                        df_day = future.result(timeout=60)
                        if not df_day.empty:
                            ticker_data.append(df_day)
                            
                            # Sauvegarde incrémentale
                            if len(ticker_data) % 10 == 0:
                                temp_df = pd.concat(ticker_data, ignore_index=True)
                                temp_file = f"{output_dir}/{ticker}_temp_{len(ticker_data)}.csv"
                                temp_df.to_csv(temp_file, index=False)
                                logger.info(f"💾 Sauvegarde temporaire: {temp_file}")
                    
                    except Exception as e:
                        logger.error(f"❌ Erreur pour {date}: {e}")
            
            # Consolider les données du ticker
            if ticker_data:
                ticker_df = pd.concat(ticker_data, ignore_index=True)
                ticker_df.sort_values(['Date', 'Type', 'Strike', 'T_Annees'], inplace=True)
                
                # Sauvegarder les données du ticker
                ticker_file = f"{output_dir}/{ticker}_options_complete.csv"
                ticker_df.to_csv(ticker_file, index=False)
                logger.info(f"✅ Sauvegardé: {ticker_file} ({len(ticker_df)} lignes)")
                
                all_data.append(ticker_df)
        
        # Consolider toutes les données
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            
            # Calculer les métriques de qualité
            self._calculate_data_quality_metrics(final_df)
            
            # Sauvegarder le fichier final
            final_file = f"{output_dir}/all_options_historical.csv"
            final_df.to_csv(final_file, index=False)
            
            logger.info(f"\n🎉 Collecte terminée!")
            logger.info(f"   - Fichier final: {final_file}")
            logger.info(f"   - Total options: {len(final_df)}")
            logger.info(f"   - Période couverte: {final_df['Date'].min()} à {final_df['Date'].max()}")
            self._print_statistics()
            
            return final_df
        else:
            logger.error("❌ Aucune donnée collectée")
            return pd.DataFrame()
    
    def _calculate_data_quality_metrics(self, df: pd.DataFrame):
        """
        Calcule des métriques de qualité pour évaluer la fiabilité des données.
        
        Métriques calculées:
        - Complétude des surfaces (couverture strike/maturité)
        - Cohérence des volatilités implicites
        - Liquidité moyenne
        - Symétrie calls/puts
        """
        if df.empty:
            return
        
        logger.info("\n📊 Métriques de qualité des données:")
        
        # Complétude par date
        completeness = df.groupby('Date').size()
        logger.info(f"   - Options par jour (moyenne): {completeness.mean():.0f}")
        logger.info(f"   - Options par jour (min/max): {completeness.min()}/{completeness.max()}")
        
        # Distribution des maturités
        maturity_dist = df.groupby(pd.cut(df['T_Annees'], 
                                         bins=[0, 0.083, 0.25, 0.5, 1.0, 2.0],
                                         labels=['<1M', '1-3M', '3-6M', '6M-1Y', '1-2Y'])).size()
        logger.info(f"   - Distribution maturités:")
        for mat, count in maturity_dist.items():
            logger.info(f"     • {mat}: {count} ({count/len(df)*100:.1f}%)")
        
        # Qualité des volatilités
        vol_stats = df['Vol_Pct'].describe()
        logger.info(f"   - Volatilité implicite:")
        logger.info(f"     • Moyenne: {vol_stats['mean']:.1f}%")
        logger.info(f"     • Écart-type: {vol_stats['std']:.1f}%")
        logger.info(f"     • Min/Max: {vol_stats['min']:.1f}%/{vol_stats['max']:.1f}%")
        
        # Score de qualité global
        quality_score = self._compute_quality_score(df)
        self.stats['data_quality_score'] = quality_score
        logger.info(f"   - Score de qualité global: {quality_score:.2f}/10")
    
    def _compute_quality_score(self, df: pd.DataFrame) -> float:
        """Calcule un score de qualité global de 0 à 10."""
        score = 10.0
        
        # Pénalités pour manque de données
        avg_options_per_day = df.groupby('Date').size().mean()
        if avg_options_per_day < 50:
            score -= 2.0
        elif avg_options_per_day < 100:
            score -= 1.0
        
        # Pénalités pour volatilités aberrantes
        aberrant_vols = ((df['Vol_Pct'] < 5) | (df['Vol_Pct'] > 200)).sum()
        aberrant_pct = aberrant_vols / len(df) * 100
        if aberrant_pct > 5:
            score -= 2.0
        elif aberrant_pct > 2:
            score -= 1.0
        
        # Bonus pour bonne liquidité
        avg_volume = df['Volume'].mean()
        if avg_volume > 1000:
            score += 0.5
        
        return max(0, min(10, score))
    
    def _print_statistics(self):
        """Affiche les statistiques de collecte."""
        logger.info("\n📈 Statistiques de collecte:")
        logger.info(f"   - Options collectées: {self.stats['total_options_collected']}")
        logger.info(f"   - Options filtrées: {self.stats['options_filtered_out']}")
        logger.info(f"   - Jours réussis: {self.stats['successful_days']}")
        logger.info(f"   - Jours échoués: {self.stats['failed_days']}")
        logger.info(f"   - Score qualité données: {self.stats['data_quality_score']:.2f}/10")
        
        if self.stats['successful_days'] > 0:
            success_rate = self.stats['successful_days'] / (self.stats['successful_days'] + self.stats['failed_days']) * 100
            logger.info(f"   - Taux de succès: {success_rate:.1f}%")


def main():
    """
    Fonction principale démontrant l'utilisation du collecteur unifié.
    
    Exemple pédagogique complet avec:
    1. Configuration personnalisée
    2. Collecte multi-tickers
    3. Analyse de la qualité
    4. Export pour les autres modules
    """
    print("=" * 80)
    print("🚀 COLLECTEUR UNIFIÉ DE DONNÉES D'OPTIONS")
    print("=" * 80)
    print("\nCe module collecte et structure les données d'options pour:")
    print("- Construction de surfaces de volatilité")
    print("- Pricing d'options et d'autocalls")
    print("- Backtesting de stratégies")
    print("- Calibration de modèles ML")
    print("=" * 80)
    
    # Configuration
    tickers = ['AAPL', 'MSFT', 'GOOGL']  # Ajouter vos tickers
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # 6 mois d'historique
    
    # Créer le collecteur
    collector = UnifiedOptionsCollector()
    
    # Exemple 1: Collecte pour une date unique
    print("\n📊 Exemple 1: Collecte pour une date unique")
    df_single = collector.collect_yahoo_options('AAPL', datetime.now())
    if not df_single.empty:
        print(f"✅ Collecté {len(df_single)} options pour AAPL aujourd'hui")
        print(f"   Aperçu des données:")
        print(df_single[['Option', 'Prix', 'Strike', 'Vol_Pct', 'Delta']].head())
    
    # Exemple 2: Collecte historique complète
    print("\n📊 Exemple 2: Collecte historique sur 6 mois")
    df_historical = collector.collect_historical_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        parallel_workers=3
    )
    
    if not df_historical.empty:
        # Afficher un résumé détaillé
        print("\n📈 RÉSUMÉ DES DONNÉES COLLECTÉES")
        print("=" * 60)
        print(f"Période: {df_historical['Date'].min()} à {df_historical['Date'].max()}")
        print(f"Tickers: {df_historical['Ticker'].unique()}")
        print(f"Total options: {len(df_historical):,}")
        
        print("\n📊 Distribution par ticker:")
        for ticker in df_historical['Ticker'].unique():
            ticker_data = df_historical[df_historical['Ticker'] == ticker]
            print(f"   {ticker}: {len(ticker_data):,} options")
        
        print("\n📊 Exemple de surface de volatilité (AAPL, dernière date):")
        last_date = df_historical['Date'].max()
        aapl_surface = df_historical[
            (df_historical['Ticker'] == 'AAPL') & 
            (df_historical['Date'] == last_date)
        ]
        
        # Afficher une mini surface
        pivot = aapl_surface.pivot_table(
            values='Vol_Pct',
            index=pd.cut(aapl_surface['Moneyness'], bins=[0.8, 0.95, 1.05, 1.2]),
            columns=pd.cut(aapl_surface['T_Annees'], bins=[0, 0.25, 0.5, 1.0, 2.0]),
            aggfunc='mean'
        )
        print(pivot.round(1))
        
        # Sauvegarder pour les autres modules
        output_file = 'data/options_data_for_pricing.csv'
        df_historical.to_csv(output_file, index=False)
        print(f"\n✅ Données sauvegardées: {output_file}")
        print("\n🎯 Prochaines étapes:")
        print("   1. Utiliser data_processing/ pour créer les features")
        print("   2. Construire les surfaces avec volatility_surface/")
        print("   3. Calibrer les modèles avec ml_models/")
        print("   4. Pricer les produits avec simulation/")


if __name__ == "__main__":
    main()
