"""
Collecteur professionnel de données d'options via Polygon.io
Auteur: Assistant Claude
Date: Mai 2025

Ce script collecte les données historiques d'options sur 6 mois via l'API Polygon.io
avec filtrage des maturités et calcul des Greeks pour les surfaces de volatilité.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import time
import os
import json
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('polygon_options_collector.log'),
        logging.StreamHandler()
    ]
)

class PolygonOptionsCollector:
    """
    Collecteur professionnel de données d'options utilisant l'API Polygon.io
    avec gestion robuste des erreurs et optimisation des performances
    """
    
    def __init__(self, api_key, rate_limit=5):
        """
        Initialise le collecteur avec la clé API Polygon.io
        
        Args:
            api_key: Clé API Polygon.io
            rate_limit: Nombre maximum de requêtes par seconde
        """
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.rate_limit = rate_limit
        self.last_request_time = 0
        
        # Paramètres de filtrage des maturités
        self.min_days_to_expiry = 7  # 1 semaine
        self.max_days_to_expiry = 730  # 2 ans
        
        # Cache pour optimiser les performances
        self.ticker_details_cache = {}
        self.option_contracts_cache = {}
        
        # Statistiques de collecte
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'options_collected': 0
        }
        
        logging.info(f"Collecteur initialisé avec rate limit: {rate_limit} req/s")
    
    def _rate_limit_request(self):
        """
        Implémente le rate limiting pour respecter les limites de l'API
        """
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        
        if time_since_last_request < min_interval:
            sleep_time = min_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_api_request(self, endpoint, params=None, max_retries=3):
        """
        Effectue une requête API avec gestion des erreurs et retry
        
        Args:
            endpoint: Point de terminaison de l'API
            params: Paramètres de la requête
            max_retries: Nombre maximum de tentatives
            
        Returns:
            dict: Réponse JSON ou None en cas d'échec
        """
        if params is None:
            params = {}
        
        params['apiKey'] = self.api_key
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(max_retries):
            try:
                self._rate_limit_request()
                self.stats['total_requests'] += 1
                
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    self.stats['successful_requests'] += 1
                    return response.json()
                elif response.status_code == 429:  # Rate limit exceeded
                    wait_time = (attempt + 1) * 5
                    logging.warning(f"Rate limit atteint, attente de {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"Erreur API {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                logging.error(f"Erreur réseau: {e}")
                
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Backoff exponentiel
        
        self.stats['failed_requests'] += 1
        return None
    
    def get_ticker_details(self, ticker):
        """
        Récupère les détails du ticker sous-jacent
        
        Args:
            ticker: Symbole du ticker
            
        Returns:
            dict: Détails du ticker
        """
        if ticker in self.ticker_details_cache:
            return self.ticker_details_cache[ticker]
        
        endpoint = f"/v3/reference/tickers/{ticker}"
        response = self._make_api_request(endpoint)
        
        if response and 'results' in response:
            self.ticker_details_cache[ticker] = response['results']
            return response['results']
        
        return None
    
    def get_options_contracts(self, underlying_ticker, as_of_date):
        """
        Récupère tous les contrats d'options pour un ticker à une date donnée
        
        Args:
            underlying_ticker: Ticker du sous-jacent
            as_of_date: Date de référence (format YYYY-MM-DD)
            
        Returns:
            list: Liste des contrats d'options
        """
        cache_key = f"{underlying_ticker}_{as_of_date}"
        if cache_key in self.option_contracts_cache:
            return self.option_contracts_cache[cache_key]
        
        endpoint = "/v3/reference/options/contracts"
        params = {
            'underlying_ticker': underlying_ticker,
            'as_of': as_of_date,
            'expired': 'false',
            'limit': 1000
        }
        
        all_contracts = []
        next_url = None
        
        while True:
            if next_url:
                response = self._make_api_request(next_url, {})
            else:
                response = self._make_api_request(endpoint, params)
            
            if not response:
                break
                
            if 'results' in response:
                contracts = response['results']
                
                # Filtrer par maturité
                filtered_contracts = self._filter_contracts_by_maturity(
                    contracts, as_of_date
                )
                all_contracts.extend(filtered_contracts)
                
            # Vérifier s'il y a d'autres pages
            if 'next_url' in response and response['next_url']:
                next_url = response['next_url']
            else:
                break
        
        self.option_contracts_cache[cache_key] = all_contracts
        logging.info(f"Trouvé {len(all_contracts)} contrats pour {underlying_ticker} le {as_of_date}")
        
        return all_contracts
    
    def _filter_contracts_by_maturity(self, contracts, as_of_date):
        """
        Filtre les contrats selon les critères de maturité (1 semaine à 2 ans)
        
        Args:
            contracts: Liste des contrats
            as_of_date: Date de référence
            
        Returns:
            list: Contrats filtrés
        """
        filtered = []
        as_of = datetime.strptime(as_of_date, '%Y-%m-%d')
        
        for contract in contracts:
            if 'expiration_date' not in contract:
                continue
                
            expiry = datetime.strptime(contract['expiration_date'], '%Y-%m-%d')
            days_to_expiry = (expiry - as_of).days
            
            if self.min_days_to_expiry <= days_to_expiry <= self.max_days_to_expiry:
                filtered.append(contract)
        
        return filtered
    
    def get_option_quote(self, option_ticker, date):
        """
        Récupère la cotation d'une option pour une date donnée
        
        Args:
            option_ticker: Ticker de l'option (format O:AAPL230616C00150000)
            date: Date de la cotation (format YYYY-MM-DD)
            
        Returns:
            dict: Données de cotation
        """
        endpoint = f"/v2/aggs/ticker/{option_ticker}/range/1/day/{date}/{date}"
        response = self._make_api_request(endpoint)
        
        if response and 'results' in response and len(response['results']) > 0:
            return response['results'][0]
        
        return None
    
    def get_underlying_price(self, ticker, date):
        """
        Récupère le prix du sous-jacent pour une date donnée
        
        Args:
            ticker: Symbole du ticker
            date: Date (format YYYY-MM-DD)
            
        Returns:
            float: Prix de clôture
        """
        endpoint = f"/v2/aggs/ticker/{ticker}/range/1/day/{date}/{date}"
        response = self._make_api_request(endpoint)
        
        if response and 'results' in response and len(response['results']) > 0:
            return response['results'][0]['c']  # Prix de clôture
        
        return None
    
    def calculate_greeks(self, option_type, S, K, T, r, sigma):
        """
        Calcule les Greeks selon le modèle Black-Scholes
        
        Args:
            option_type: 'call' ou 'put'
            S: Prix du sous-jacent
            K: Prix d'exercice
            T: Temps jusqu'à expiration (années)
            r: Taux sans risque
            sigma: Volatilité implicite
            
        Returns:
            dict: Greeks calculés
        """
        # Protection contre les valeurs invalides
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
        
        # Calculs Black-Scholes
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        # Delta
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        
        # Gamma
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        if option_type.lower() == 'call':
            term2 = -r * K * np.exp(-r*T) * norm.cdf(d2)
        else:
            term2 = r * K * np.exp(-r*T) * norm.cdf(-d2)
        theta = (term1 + term2) / 365  # Theta journalier
        
        # Vega
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        return {
            'delta': round(delta, 4),
            'gamma': round(gamma, 6),
            'theta': round(theta, 4),
            'vega': round(vega, 4)
        }
    
    def collect_daily_options_data(self, ticker, collection_date, risk_free_rate=0.05):
        """
        Collecte toutes les données d'options pour un ticker à une date donnée
        
        Args:
            ticker: Symbole du ticker
            collection_date: Date de collecte (format YYYY-MM-DD)
            risk_free_rate: Taux sans risque pour calcul des Greeks
            
        Returns:
            pd.DataFrame: Données d'options collectées
        """
        logging.info(f"Collecte des données pour {ticker} le {collection_date}")
        
        # Récupérer le prix du sous-jacent
        underlying_price = self.get_underlying_price(ticker, collection_date)
        if not underlying_price:
            logging.warning(f"Impossible de récupérer le prix de {ticker} pour {collection_date}")
            return pd.DataFrame()
        
        # Récupérer tous les contrats d'options
        contracts = self.get_options_contracts(ticker, collection_date)
        if not contracts:
            logging.warning(f"Aucun contrat trouvé pour {ticker} le {collection_date}")
            return pd.DataFrame()
        
        options_data = []
        
        # Traiter chaque contrat
        for i, contract in enumerate(contracts):
            if i % 50 == 0:
                logging.info(f"Traitement {i}/{len(contracts)} contrats...")
            
            try:
                # Extraire les informations du contrat
                option_ticker = contract['ticker']
                strike = contract['strike_price']
                expiry_date = contract['expiration_date']
                option_type = contract['contract_type']
                
                # Calculer le temps jusqu'à expiration
                expiry = datetime.strptime(expiry_date, '%Y-%m-%d')
                collection = datetime.strptime(collection_date, '%Y-%m-%d')
                days_to_expiry = (expiry - collection).days
                years_to_expiry = days_to_expiry / 365.25
                
                # Récupérer la cotation de l'option
                quote = self.get_option_quote(option_ticker, collection_date)
                if not quote:
                    continue
                
                # Extraire les données de cotation
                close_price = quote.get('c', 0)
                volume = quote.get('v', 0)
                vwap = quote.get('vw', close_price)  # Volume weighted average price
                
                # Calculer la volatilité implicite approximative
                # Note: Polygon.io ne fournit pas toujours l'IV directement
                moneyness = strike / underlying_price
                base_vol = 0.25  # Volatilité de base
                smile_effect = 0.1 * abs(np.log(moneyness)) ** 1.5
                time_adjustment = np.sqrt(30 / max(days_to_expiry, 1))
                impl_vol = (base_vol + smile_effect) * time_adjustment
                
                # Si Polygon fournit l'IV, l'utiliser
                if 'implied_volatility' in quote:
                    impl_vol = quote['implied_volatility']
                
                # Calculer les Greeks
                greeks = self.calculate_greeks(
                    option_type=option_type,
                    S=underlying_price,
                    K=strike,
                    T=years_to_expiry,
                    r=risk_free_rate,
                    sigma=impl_vol
                )
                
                # Créer l'enregistrement de données
                option_data = {
                    'dataDate': collection_date,
                    'ticker': ticker,
                    'optionTicker': option_ticker,
                    'optionType': option_type.upper(),
                    'strike': strike,
                    'expirationDate': expiry_date,
                    'daysToExpiration': days_to_expiry,
                    'yearsToExpiration': round(years_to_expiry, 4),
                    'lastPrice': close_price,
                    'volume': volume,
                    'vwap': vwap,
                    'impliedVolatility': impl_vol,
                    'impliedVolatilityPct': round(impl_vol * 100, 2),
                    'underlyingPrice': underlying_price,
                    'moneyness': round(moneyness, 4),
                    'delta': greeks['delta'],
                    'gamma': greeks['gamma'],
                    'theta': greeks['theta'],
                    'vega': greeks['vega'],
                    'inTheMoney': (option_type.lower() == 'call' and underlying_price > strike) or 
                                  (option_type.lower() == 'put' and underlying_price < strike)
                }
                
                options_data.append(option_data)
                self.stats['options_collected'] += 1
                
            except Exception as e:
                logging.error(f"Erreur traitement contrat {option_ticker}: {e}")
                continue
        
        df = pd.DataFrame(options_data)
        logging.info(f"Collecté {len(df)} options pour {ticker} le {collection_date}")
        
        return df
    
    def collect_historical_data(self, ticker, days_back=180, end_date=None, 
                              parallel_days=5, save_intermediate=True):
        """
        Collecte les données historiques d'options pour plusieurs jours
        
        Args:
            ticker: Symbole du ticker
            days_back: Nombre de jours d'historique (défaut: 180 pour 6 mois)
            end_date: Date de fin (défaut: aujourd'hui)
            parallel_days: Nombre de jours à traiter en parallèle
            save_intermediate: Sauvegarder les résultats intermédiaires
            
        Returns:
            pd.DataFrame: Toutes les données historiques
        """
        logging.info(f"Début de la collecte historique pour {ticker} sur {days_back} jours")
        
        # Déterminer la période de collecte
        if end_date is None:
            end_date = datetime.now().date()
        else:
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        start_date = end_date - timedelta(days=days_back)
        
        # Générer la liste des dates de trading (exclure weekends)
        trading_days = []
        current_date = start_date
        
        while current_date <= end_date:
            # Exclure les weekends (samedi=5, dimanche=6)
            if current_date.weekday() < 5:
                trading_days.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
        
        logging.info(f"Collecte prévue pour {len(trading_days)} jours de trading")
        
        all_data = []
        
        # Créer le répertoire pour les sauvegardes
        output_dir = f"polygon_options_data/{ticker}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Traiter les jours par lots pour optimiser
        with ThreadPoolExecutor(max_workers=parallel_days) as executor:
            for i in range(0, len(trading_days), parallel_days):
                batch_days = trading_days[i:i+parallel_days]
                futures = []
                
                for day in batch_days:
                    future = executor.submit(
                        self.collect_daily_options_data, 
                        ticker, 
                        day
                    )
                    futures.append((day, future))
                
                # Attendre que le lot soit terminé
                for day, future in futures:
                    try:
                        df_day = future.result(timeout=300)  # 5 minutes timeout
                        
                        if not df_day.empty:
                            all_data.append(df_day)
                            
                            # Sauvegarder les données quotidiennes
                            if save_intermediate:
                                day_file = f"{output_dir}/{ticker}_options_{day}.csv"
                                df_day.to_csv(day_file, index=False)
                                logging.info(f"Sauvegardé: {day_file}")
                        
                    except Exception as e:
                        logging.error(f"Erreur collecte {day}: {e}")
                
                # Pause entre les lots
                time.sleep(2)
                
                # Afficher les statistiques
                if (i + parallel_days) % 20 == 0:
                    self._print_statistics()
        
        # Combiner toutes les données
        if all_data:
            df_combined = pd.concat(all_data, ignore_index=True)
            df_combined.sort_values(['dataDate', 'optionType', 'strike', 'expirationDate'], 
                                  inplace=True)
            
            # Sauvegarder le fichier combiné
            output_file = f"{output_dir}/{ticker}_options_historical_{days_back}d.csv"
            df_combined.to_csv(output_file, index=False)
            
            logging.info(f"✅ Collecte terminée: {len(df_combined)} options")
            logging.info(f"📁 Fichier final: {output_file}")
            
            self._print_statistics()
            
            return df_combined
        else:
            logging.warning("Aucune donnée collectée")
            return pd.DataFrame()
    
    def _print_statistics(self):
        """
        Affiche les statistiques de collecte
        """
        logging.info("📊 Statistiques de collecte:")
        logging.info(f"   - Requêtes totales: {self.stats['total_requests']}")
        logging.info(f"   - Requêtes réussies: {self.stats['successful_requests']}")
        logging.info(f"   - Requêtes échouées: {self.stats['failed_requests']}")
        logging.info(f"   - Options collectées: {self.stats['options_collected']}")
        
        if self.stats['total_requests'] > 0:
            success_rate = (self.stats['successful_requests'] / self.stats['total_requests']) * 100
            logging.info(f"   - Taux de succès: {success_rate:.1f}%")
    
    def validate_data_quality(self, df):
        """
        Valide et nettoie les données collectées
        
        Args:
            df: DataFrame des données d'options
            
        Returns:
            pd.DataFrame: Données nettoyées et validées
        """
        initial_count = len(df)
        
        # Filtrer les données invalides
        df = df[df['lastPrice'] > 0.01]  # Prix minimum
        df = df[df['impliedVolatilityPct'] > 1]  # Vol minimum 1%
        df = df[df['impliedVolatilityPct'] < 500]  # Vol maximum 500%
        df = df[df['volume'] >= 0]  # Volume non négatif
        df = df[df['yearsToExpiration'] > 0]  # Pas d'options expirées
        
        # Calculer des métriques de qualité
        removed_count = initial_count - len(df)
        removal_rate = (removed_count / initial_count) * 100 if initial_count > 0 else 0
        
        logging.info(f"🧹 Nettoyage des données:")
        logging.info(f"   - Enregistrements initiaux: {initial_count}")
        logging.info(f"   - Enregistrements supprimés: {removed_count} ({removal_rate:.1f}%)")
        logging.info(f"   - Enregistrements finaux: {len(df)}")
        
        # Ajouter des colonnes utiles pour l'analyse
        df['spreadPct'] = ((df['strike'] - df['underlyingPrice']) / df['underlyingPrice']) * 100
        df['daysToExpiration'] = df['daysToExpiration'].astype(int)
        
        return df

# Fonction principale d'utilisation
def main():
    """
    Fonction principale pour démontrer l'utilisation du collecteur
    """
    # Configuration
    API_KEY = "dEslJ8Po7QyXvQ5E_BpMoq0LTpcs8u66"
    TICKER = "AAPL"
    DAYS_BACK = 180  # 6 mois
    
    print("🚀 Démarrage du collecteur de données d'options Polygon.io")
    print("=" * 70)
    
    # Initialiser le collecteur
    collector = PolygonOptionsCollector(API_KEY)
    
    # Collecter les données historiques
    df_historical = collector.collect_historical_data(
        ticker=TICKER,
        days_back=DAYS_BACK,
        parallel_days=3,  # Traiter 3 jours en parallèle
        save_intermediate=True
    )
    
    if not df_historical.empty:
        # Valider et nettoyer les données
        df_clean = collector.validate_data_quality(df_historical)
        
        # Afficher un résumé
        print("\n📊 Résumé des données collectées:")
        print(f"   - Période: {df_clean['dataDate'].min()} à {df_clean['dataDate'].max()}")
        print(f"   - Nombre total d'options: {len(df_clean)}")
        print(f"   - Dates uniques: {df_clean['dataDate'].nunique()}")
        print(f"   - Dates d'expiration uniques: {df_clean['expirationDate'].nunique()}")
        print(f"   - Strikes uniques: {df_clean['strike'].nunique()}")
        
        # Statistiques par type d'option
        print("\n📈 Répartition par type:")
        print(df_clean['optionType'].value_counts())
        
        # Exemple d'utilisation avec vos scripts
        print("\n✅ Les données sont prêtes pour vos scripts de surface de volatilité!")
        print(f"📁 Fichier principal: polygon_options_data/{TICKER}/{TICKER}_options_historical_{DAYS_BACK}d.csv")
        
    else:
        print("❌ Aucune donnée collectée")

if __name__ == "__main__":
    main()
