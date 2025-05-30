"""
Collecteur Unifié de Données d'Options avec YFinance - Version Production
Auteur: Assistant Claude
Date: Mai 2025

Ce module collecte des données RÉELLES d'options depuis Yahoo Finance
et génère des données structurées pour l'analyse de volatilité.

FONCTIONNALITÉS:
- Collecte temps réel depuis Yahoo Finance
- Structure unifiée pour surfaces de volatilité
- Calcul automatique des Greeks
- Filtrage intelligent par liquidité et maturité
- Support pour collecte historique simulée
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import yfinance as yf
import time
import os
import json
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unified_collection.log'),
        logging.StreamHandler()
    ]
)

warnings.filterwarnings('ignore')

class UnifiedYFinanceCollector:
    """
    Collecteur unifié utilisant Yahoo Finance avec toutes les fonctionnalités.
    
    ARCHITECTURE:
    - Collecte multi-threading pour performance
    - Calcul des Greeks en temps réel
    - Validation et nettoyage automatiques
    - Structure optimisée pour surfaces de volatilité
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialise le collecteur.
        
        Args:
            risk_free_rate: Taux sans risque (défaut: 5%)
        """
        self.risk_free_rate = risk_free_rate
        self.min_days_to_expiry = 7      # 1 semaine minimum
        self.max_days_to_expiry = 730    # 2 ans maximum
        self.min_volume = 1              # Volume minimal
        self.min_open_interest = 10      # OI minimal
        self.max_moneyness = 2.0         # 200% maximum
        self.min_moneyness = 0.5         # 50% minimum
        self._cache = {}
        
        logging.info("🚀 Collecteur Unifié YFinance initialisé")
        logging.info(f"📊 Paramètres: r={risk_free_rate:.2%}, "
                    f"maturité={self.min_days_to_expiry}-{self.max_days_to_expiry}j")

    def collect_options_data(self, ticker: str, 
                           collection_date: Optional[date] = None,
                           max_expirations: Optional[int] = None,
                           include_greeks: bool = True) -> pd.DataFrame:
        """
        Collecte les données d'options pour un ticker donné.
        
        Args:
            ticker: Symbole du sous-jacent
            collection_date: Date de collecte (défaut: aujourd'hui)
            max_expirations: Limite le nombre d'expirations (None = toutes)
            include_greeks: Calculer les Greeks
            
        Returns:
            DataFrame structuré pour les surfaces de volatilité
        """
        if collection_date is None:
            collection_date = date.today()
            
        logging.info(f"📊 === COLLECTE POUR {ticker} le {collection_date} ===")
        start_time = time.time()
        
        # 1. Obtenir les informations du sous-jacent à la date spécifique
        stock_info = self._get_stock_info(ticker, collection_date)
        if not stock_info:
            return pd.DataFrame()
        
        spot_price = stock_info['price']
        logging.info(f"💰 Prix spot au {collection_date}: ${spot_price:.2f}")
        
        # 2. Collecter toutes les chaînes d'options
        all_options = self._collect_all_option_chains(
            ticker, spot_price, collection_date, max_expirations
        )
        
        if all_options.empty:
            logging.error("❌ Aucune donnée d'options collectée")
            return pd.DataFrame()
        
        logging.info(f"📈 {len(all_options)} options collectées avant filtrage")
        
        # 3. Filtrer par qualité
        filtered_options = self._filter_options(all_options, collection_date)
        logging.info(f"✅ {len(filtered_options)} options après filtrage")
        
        # 4. Calculer les Greeks si demandé
        if include_greeks and not filtered_options.empty:
            filtered_options = self._calculate_all_greeks(filtered_options, spot_price)
            logging.info("🧮 Greeks calculés")
        
        # 5. Structurer pour la surface de volatilité
        structured_data = self._structure_for_surface(
            filtered_options, collection_date, spot_price, ticker
        )
        
        elapsed = time.time() - start_time
        logging.info(f"⏱️  Collecte terminée en {elapsed:.1f}s")
        logging.info(f"📊 Résultat: {len(structured_data)} options structurées")
        
        return structured_data

    def _get_stock_info(self, ticker: str, target_date: date) -> Optional[Dict]:
        """
        Récupère les informations du sous-jacent à une date spécifique.
        
        Args:
            ticker: Symbole du sous-jacent
            target_date: Date pour laquelle récupérer le prix
            
        Returns:
            Dict avec le prix et autres infos à la date cible
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Si c'est aujourd'hui, utiliser le prix actuel
            if target_date == date.today():
                try:
                    info = stock.info
                    price = info.get('currentPrice') or info.get('regularMarketPrice')
                    if price:
                        return {
                            'price': price,
                            'volume': info.get('volume', 0),
                            'market_cap': info.get('marketCap', 0),
                            'dividend_yield': info.get('dividendYield', 0),
                            'historical': False
                        }
                except:
                    pass
            
            # Récupérer l'historique pour la date spécifique
            # Ajouter quelques jours de marge pour s'assurer d'avoir la donnée
            start_date = target_date - timedelta(days=10)
            end_date = target_date + timedelta(days=1)
            
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                logging.error(f"❌ Pas de données historiques pour {ticker} au {target_date}")
                return None
            
            # Trouver le prix le plus proche de la date cible
            # Convertir l'index en dates pour la comparaison
            hist.index = pd.to_datetime(hist.index).date
            
            # Si la date exacte existe
            if target_date in hist.index:
                close_price = hist.loc[target_date, 'Close']
                volume = hist.loc[target_date, 'Volume']
            else:
                # Sinon prendre la date la plus proche avant
                valid_dates = [d for d in hist.index if d <= target_date]
                if not valid_dates:
                    valid_dates = hist.index.tolist()
                
                closest_date = max(valid_dates)
                close_price = hist.loc[closest_date, 'Close']
                volume = hist.loc[closest_date, 'Volume']
                logging.info(f"📅 Utilisation du prix du {closest_date} (plus proche de {target_date})")
            
            return {
                'price': float(close_price),
                'volume': int(volume),
                'market_cap': 0,
                'dividend_yield': 0,
                'historical': True,
                'price_date': str(closest_date) if 'closest_date' in locals() else str(target_date)
            }
            
        except Exception as e:
            logging.error(f"❌ Erreur récupération {ticker} pour {target_date}: {e}")
            return None

    def _collect_all_option_chains(self, ticker: str, spot_price: float,
                                 collection_date: date, 
                                 max_expirations: Optional[int]) -> pd.DataFrame:
        """Collecte toutes les chaînes d'options disponibles."""
        stock = yf.Ticker(ticker)
        
        try:
            # Obtenir toutes les expirations
            expirations = stock.options
            if max_expirations:
                expirations = expirations[:max_expirations]
                
            logging.info(f"📅 {len(expirations)} expirations à traiter")
            
        except Exception as e:
            logging.error(f"❌ Erreur récupération expirations: {e}")
            return pd.DataFrame()
        
        # Utiliser ThreadPoolExecutor pour paralléliser
        all_chains = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Soumettre toutes les tâches
            future_to_exp = {
                executor.submit(
                    self._process_single_expiration, 
                    stock, exp_date, ticker, spot_price, collection_date
                ): exp_date 
                for exp_date in expirations
            }
            
            # Collecter les résultats
            for future in as_completed(future_to_exp):
                exp_date = future_to_exp[future]
                try:
                    result = future.result()
                    if result is not None and not result.empty:
                        all_chains.append(result)
                except Exception as e:
                    logging.warning(f"⚠️  Erreur pour {exp_date}: {e}")
        
        if all_chains:
            return pd.concat(all_chains, ignore_index=True)
        else:
            return pd.DataFrame()

    def _process_single_expiration(self, stock: yf.Ticker, exp_date: str,
                                 ticker: str, spot_price: float,
                                 collection_date: date) -> Optional[pd.DataFrame]:
        """Traite une seule date d'expiration."""
        try:
            # Calculer les jours jusqu'à expiration
            exp_datetime = pd.to_datetime(exp_date)
            days_to_exp = (exp_datetime.date() - collection_date).days
            
            # Filtrer par maturité
            if not (self.min_days_to_expiry <= days_to_exp <= self.max_days_to_expiry):
                return None
            
            # Récupérer la chaîne
            chain = stock.option_chain(exp_date)
            
            # Traiter calls et puts
            calls = self._process_option_type(
                chain.calls, 'CALL', ticker, exp_date, days_to_exp, spot_price
            )
            puts = self._process_option_type(
                chain.puts, 'PUT', ticker, exp_date, days_to_exp, spot_price
            )
            
            # Combiner
            if not calls.empty or not puts.empty:
                return pd.concat([calls, puts], ignore_index=True)
            else:
                return None
                
        except Exception as e:
            logging.debug(f"Erreur expiration {exp_date}: {e}")
            return None

    def _process_option_type(self, options_df: pd.DataFrame, option_type: str,
                           ticker: str, exp_date: str, days_to_exp: int,
                           spot_price: float) -> pd.DataFrame:
        """Traite un type d'options (CALL ou PUT)."""
        if options_df.empty:
            return pd.DataFrame()
        
        df = options_df.copy()
        
        # Ajouter les métadonnées
        df['ticker'] = ticker
        df['optionType'] = option_type
        df['expirationDate'] = exp_date
        df['daysToExpiration'] = days_to_exp
        df['yearsToExpiration'] = days_to_exp / 365.25
        df['spotPrice'] = spot_price
        
        # Calculer la moneyness
        df['moneyness'] = df['strike'] / spot_price
        df['moneynessPct'] = (df['moneyness'] * 100).round(0).astype(int)
        
        # Prix amélioré
        df['mid_price'] = df.apply(
            lambda row: (row['bid'] + row['ask']) / 2 
            if row['bid'] > 0 and row['ask'] > 0 
            else row['lastPrice'], 
            axis=1
        )
        
        # Nettoyer la volatilité implicite
        df['impliedVolatility'] = df['impliedVolatility'].apply(
            lambda x: self._clean_implied_volatility(x, option_type)
        )
        
        # ITM/OTM
        if option_type == 'CALL':
            df['inTheMoney'] = df['strike'] < spot_price
        else:
            df['inTheMoney'] = df['strike'] > spot_price
        
        return df

    def _clean_implied_volatility(self, iv: float, option_type: str) -> float:
        """Nettoie et normalise la volatilité implicite."""
        # Gérer les valeurs manquantes
        if pd.isna(iv) or iv <= 0:
            return 25.0  # Valeur par défaut
        
        # Convertir en pourcentage si nécessaire
        if iv < 5:  # Probablement en décimal
            iv = iv * 100
        
        # Limiter aux valeurs réalistes
        return max(5.0, min(iv, 200.0))

    def _filter_options(self, options_df: pd.DataFrame, 
                       collection_date: date) -> pd.DataFrame:
        """Filtre les options selon les critères de qualité."""
        initial_count = len(options_df)
        
        # Filtres de base
        df = options_df[
            (options_df['mid_price'] >= 0.05) &
            (options_df['impliedVolatility'] >= 5) &
            (options_df['impliedVolatility'] <= 200) &
            (options_df['moneyness'] >= self.min_moneyness) &
            (options_df['moneyness'] <= self.max_moneyness) &
            ((options_df['volume'] >= self.min_volume) | 
             (options_df['openInterest'] >= self.min_open_interest))
        ].copy()
        
        # Filtre spread bid-ask
        df = df[
            df.apply(
                lambda row: True if row['bid'] == 0 or row['ask'] == 0
                else (row['ask'] - row['bid']) / row['mid_price'] < 0.5,
                axis=1
            )
        ]
        
        removed = initial_count - len(df)
        logging.info(f"🧹 Filtrage: {removed} options supprimées "
                    f"({removed/initial_count*100:.1f}%)")
        
        return df

    def _calculate_all_greeks(self, options_df: pd.DataFrame, 
                            spot_price: float) -> pd.DataFrame:
        """Calcule tous les Greeks pour les options."""
        df = options_df.copy()
        
        # Initialiser les colonnes
        for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
            df[greek] = 0.0
        
        # Calculer pour chaque option
        for idx, row in df.iterrows():
            greeks = self._calculate_greeks(
                option_type=row['optionType'],
                S=spot_price,
                K=row['strike'],
                T=row['yearsToExpiration'],
                r=self.risk_free_rate,
                sigma=row['impliedVolatility'] / 100
            )
            
            for greek, value in greeks.items():
                df.at[idx, greek] = value
        
        return df

    def _calculate_greeks(self, option_type: str, S: float, K: float,
                        T: float, r: float, sigma: float) -> Dict[str, float]:
        """Calcule les Greeks selon Black-Scholes."""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        try:
            # Black-Scholes
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            # Delta
            if option_type == 'CALL':
                delta = norm.cdf(d1)
            else:
                delta = norm.cdf(d1) - 1
            
            # Gamma
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            
            # Theta
            term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            if option_type == 'CALL':
                term2 = -r * K * np.exp(-r*T) * norm.cdf(d2)
            else:
                term2 = r * K * np.exp(-r*T) * norm.cdf(-d2)
            theta = (term1 + term2) / 365
            
            # Vega
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100
            
            # Rho
            if option_type == 'CALL':
                rho = K * T * np.exp(-r*T) * norm.cdf(d2) / 100
            else:
                rho = -K * T * np.exp(-r*T) * norm.cdf(-d2) / 100
            
            return {
                'delta': round(delta, 4),
                'gamma': round(gamma, 6),
                'theta': round(theta, 4),
                'vega': round(vega, 4),
                'rho': round(rho, 4)
            }
            
        except:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

    def _structure_for_surface(self, options_df: pd.DataFrame,
                             collection_date: date, spot_price: float,
                             ticker: str) -> pd.DataFrame:
        """Structure les données pour les surfaces de volatilité."""
        structured_data = []
        surface_id = f"surface_vol_{collection_date.strftime('%Y%m%d')}"
        
        for _, opt in options_df.iterrows():
            # Label de maturité
            years = opt['yearsToExpiration']
            if years <= 0.08:  # ~1 mois
                maturity_label = "1M"
            elif years <= 0.25:  # ~3 mois
                maturity_label = "3M"
            elif years <= 0.5:  # ~6 mois
                maturity_label = "6M"
            elif years <= 1.0:
                maturity_label = "1Y"
            else:
                maturity_label = f"{years:.1f}Y"
            
            option_label = f"{opt['moneynessPct']}% {maturity_label} {opt['optionType']}"
            
            # Enregistrement structuré
            record = {
                # Identifiants
                'dataDate': collection_date.strftime('%Y-%m-%d'),
                'surfaceId': surface_id,
                'ticker': ticker,
                'optionLabel': option_label,
                
                # Données de base
                'optionType': opt['optionType'],
                'strike': opt['strike'],
                'expirationDate': opt['expirationDate'],
                'daysToExpiration': opt['daysToExpiration'],
                'yearsToExpiration': round(opt['yearsToExpiration'], 4),
                
                # Pricing
                'spotPrice': spot_price,
                'optionPrice': opt['mid_price'],
                'bid': opt['bid'],
                'ask': opt['ask'],
                'lastPrice': opt['lastPrice'],
                
                # Moneyness
                'moneyness': round(opt['moneyness'], 4),
                'moneynessPct': opt['moneynessPct'],
                'inTheMoney': opt['inTheMoney'],
                
                # Volatilité
                'impliedVolatility': opt['impliedVolatility'],
                
                # Greeks
                'delta': opt.get('delta', 0),
                'gamma': opt.get('gamma', 0),
                'theta': opt.get('theta', 0),
                'vega': opt.get('vega', 0),
                'rho': opt.get('rho', 0),
                
                # Liquidité
                'volume': opt['volume'],
                'openInterest': opt['openInterest']
            }
            
            structured_data.append(record)
        
        df = pd.DataFrame(structured_data)
        
        # Trier pour faciliter l'analyse
        df.sort_values(
            ['optionType', 'yearsToExpiration', 'moneynessPct'],
            inplace=True
        )
        
        return df

    def collect_historical_data(self, ticker: str, days_back: int = 1,
                              save_intermediate: bool = True) -> pd.DataFrame:
        """
        Collecte les données sur plusieurs jours (simulation historique).
        
        Note: Yahoo Finance ne fournit que les données actuelles,
        donc on simule l'historique en collectant plusieurs fois.
        """
        logging.info(f"🚀 Collecte historique pour {ticker}")
        logging.info(f"📅 Période: {days_back} jour(s)")
        
        # Créer le répertoire
        output_dir = f"data/{ticker}"
        os.makedirs(output_dir, exist_ok=True)
        
        all_data = []
        
        # Pour chaque jour (simulation)
        for day_offset in range(days_back):
            collection_date = date.today() - timedelta(days=day_offset)
            
            # Skip weekends
            if collection_date.weekday() >= 5:
                continue
                
            logging.info(f"📊 Collecte pour {collection_date}")
            
            try:
                # Collecter les données
                daily_data = self.collect_options_data(
                    ticker=ticker,
                    collection_date=collection_date,
                    max_expirations=10  # Limiter pour la performance
                )
                
                if not daily_data.empty:
                    all_data.append(daily_data)
                    
                    # Sauvegarder si demandé
                    if save_intermediate:
                        filename = f"{output_dir}/{ticker}_options_{collection_date.strftime('%Y%m%d')}.csv"
                        daily_data.to_csv(filename, index=False)
                        logging.info(f"💾 Sauvegardé: {filename}")
                
                # Pause entre les collectes
                if day_offset < days_back - 1:
                    time.sleep(2)
                    
            except Exception as e:
                logging.error(f"❌ Erreur pour {collection_date}: {e}")
        
        # Combiner toutes les données
        if all_data:
            df_combined = pd.concat(all_data, ignore_index=True)
            
            # Sauvegarder le fichier combiné
            output_file = f"{output_dir}/{ticker}_options_combined_{days_back}d.csv"
            df_combined.to_csv(output_file, index=False)
            
            logging.info(f"✅ Collecte terminée!")
            logging.info(f"📊 Total: {len(df_combined)} options")
            logging.info(f"💾 Fichier: {output_file}")
            
            return df_combined
        else:
            return pd.DataFrame()

    def analyze_surface(self, df: pd.DataFrame, create_plots: bool = True) -> Dict:
        """Analyse la surface de volatilité."""
        if df.empty:
            return {}
        
        analysis = {
            'stats': {
                'total_options': len(df),
                'calls': len(df[df['optionType'] == 'CALL']),
                'puts': len(df[df['optionType'] == 'PUT']),
                'avg_iv': df['impliedVolatility'].mean(),
                'total_volume': df['volume'].sum(),
                'dates': df['dataDate'].nunique()
            }
        }
        
        # Analyse par maturité
        analysis['by_maturity'] = df.groupby('daysToExpiration').agg({
            'impliedVolatility': ['mean', 'std'],
            'volume': 'sum',
            'openInterest': 'sum'
        }).round(2)
        
        # Smile par moneyness
        analysis['smile'] = df.groupby(['moneynessPct', 'optionType'])['impliedVolatility'].mean().round(2)
        
        # Afficher le résumé
        print("\n" + "="*60)
        print("📊 ANALYSE DE LA SURFACE DE VOLATILITÉ")
        print("="*60)
        print(f"📈 Total options: {analysis['stats']['total_options']}")
        print(f"   └─ Calls: {analysis['stats']['calls']} | Puts: {analysis['stats']['puts']}")
        print(f"📊 Volatilité moyenne: {analysis['stats']['avg_iv']:.1f}%")
        print(f"💹 Volume total: {analysis['stats']['total_volume']:,}")
        print(f"📅 Dates: {analysis['stats']['dates']}")
        
        # Créer les graphiques si demandé
        if create_plots:
            self._create_surface_plots(df)
        
        return analysis

    def _create_surface_plots(self, df: pd.DataFrame):
        """Crée les graphiques de visualisation."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Surface de Volatilité - Analyse', fontsize=16)
            
            # 1. Term structure
            term_structure = df.groupby('daysToExpiration')['impliedVolatility'].mean()
            axes[0, 0].plot(term_structure.index, term_structure.values, 'b-o')
            axes[0, 0].set_title('Structure par Terme')
            axes[0, 0].set_xlabel('Jours jusqu\'à expiration')
            axes[0, 0].set_ylabel('Vol Implicite (%)')
            axes[0, 0].grid(True)
            
            # 2. Smile
            for opt_type in ['CALL', 'PUT']:
                smile_data = df[df['optionType'] == opt_type].groupby('moneynessPct')['impliedVolatility'].mean()
                axes[0, 1].plot(smile_data.index, smile_data.values, 'o-', label=opt_type)
            axes[0, 1].set_title('Smile de Volatilité')
            axes[0, 1].set_xlabel('Moneyness (%)')
            axes[0, 1].set_ylabel('Vol Implicite (%)')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # 3. Distribution des volumes
            axes[1, 0].hist(df['volume'], bins=50, alpha=0.7)
            axes[1, 0].set_title('Distribution des Volumes')
            axes[1, 0].set_xlabel('Volume')
            axes[1, 0].set_ylabel('Fréquence')
            axes[1, 0].set_yscale('log')
            
            # 4. Heatmap
            pivot = df.pivot_table(
                values='impliedVolatility',
                index='moneynessPct',
                columns='daysToExpiration',
                aggfunc='mean'
            )
            sns.heatmap(pivot, ax=axes[1, 1], cmap='viridis', 
                       cbar_kws={'label': 'Vol Implicite (%)'})
            axes[1, 1].set_title('Surface de Volatilité')
            
            plt.tight_layout()
            plt.savefig('volatility_surface.png', dpi=300)
            plt.show()
            
            logging.info("📊 Graphiques sauvegardés: volatility_surface.png")
            
        except Exception as e:
            logging.warning(f"⚠️  Erreur création graphiques: {e}")


def main():
    """Fonction principale de démonstration."""
    print("="*80)
    print("🚀 COLLECTEUR UNIFIÉ YFINANCE - DONNÉES RÉELLES")
    print("="*80)
    
    # Configuration
    TICKER = "AAPL"  # Changer selon vos besoins
    
    # Créer le collecteur
    collector = UnifiedYFinanceCollector(risk_free_rate=0.05)
    
    # 1. Collecte simple pour aujourd'hui
    print(f"\n📊 Collecte des options pour {TICKER}")
    print("-" * 50)
    
    df_today = collector.collect_options_data(
        ticker=TICKER,
        max_expirations=60,  # Limiter pour la démo
        include_greeks=True
    )
    
    if not df_today.empty:
        # Afficher un aperçu
        print(f"\n✅ {len(df_today)} options collectées")
        print("\n📈 APERÇU (Top 10 par volume):")
        print("-" * 80)
        
        top_options = df_today.nlargest(10, 'volume')[
            ['optionLabel', 'optionPrice', 'impliedVolatility', 
             'delta', 'volume', 'openInterest']
        ]
        print(top_options.to_string(index=False))
        
        # Analyser la surface
        analysis = collector.analyze_surface(df_today, create_plots=False)
        
        # 2. Collecte historique (optionnel)
        print(f"\n📅 Collecte historique sur 180 jours")
        print("-" * 50)
        
        df_historical = collector.collect_historical_data(
            ticker=TICKER,
            days_back=180,
            save_intermediate=False
        )
        
        if not df_historical.empty:
            print(f"\n📊 Données historiques collectées:")
            print(f"   - Dates: {df_historical['dataDate'].unique()}")
            print(f"   - Options: {len(df_historical)}")
            
            # Distribution par type et maturité
            summary = df_historical.groupby(['optionType', 'daysToExpiration']).size()
            print("\n📊 Distribution par type et maturité:")
            print(summary.head(10))
    
    print("\n✅ Démonstration terminée!")
    print("💡 Les données sont prêtes pour l'analyse des surfaces de volatilité")


if __name__ == "__main__":
    main()
