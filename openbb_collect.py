import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import norm

# Configuration du logging (journalisation)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('openbb_tradier_options_collector.log'),
        logging.StreamHandler()
    ]
)

class OpenBBTradierOptionsCollector:
    """
    Collecteur de données d'options utilisant le SDK OpenBB (source Tradier).
    Ce collecteur récupère les chaînes d'options historiques sur 6 mois, 
    filtre par maturité et calcule les Greeks via Black-Scholes.
    """
    
    def __init__(self, tradier_token=None, rate_limit=5):
        """
        Initialise le collecteur avec la clé API Tradier et le rate limit.
        
        Args:
            tradier_token (str): Clé API Tradier (jeton d'accès). Si None, on suppose qu'elle est déjà configurée.
            rate_limit (int): Nombre maximum de requêtes par seconde.
        """
        # Si un token est fourni, le configurer (via variable d'environnement attendue par OpenBB)
        if tradier_token:
            os.environ["OPENBB_API_TRADIER_TOKEN"] = tradier_token
        
        # Charger le SDK OpenBB
        try:
            from openbb_terminal.sdk import openbb
            self.openbb = openbb
        except ImportError as e:
            logging.error("Erreur import OpenBB SDK: assurez-vous que la bibliothèque OpenBB est installée.")
            raise e
        
        self.rate_limit = rate_limit
        self.last_request_time = 0
        
        # Paramètres de filtrage des maturités (en jours)
        self.min_days_to_expiry = 7    # au moins 1 semaine avant expiration
        self.max_days_to_expiry = 730  # au plus 2 ans avant expiration
        
        # Statistiques de collecte
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'options_collected': 0
        }
        
        logging.info(f"Collecteur OpenBB Tradier initialisé (rate limit: {rate_limit} req/s)")
    
    def _rate_limit_request(self):
        """
        Applique un délai pour respecter le taux maximum de requêtes.
        """
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)
        # Mettre à jour l'heure de la dernière requête
        self.last_request_time = time.time()
    
    def get_underlying_price(self, ticker, as_of_date):
        """
        Récupère le prix de clôture du sous-jacent pour une date donnée.
        
        Args:
            ticker (str): Symbole du ticker (sous-jacent).
            as_of_date (str): Date au format 'YYYY-MM-DD'.
        
        Returns:
            float: Prix de clôture ajusté du sous-jacent ce jour-là, ou None en cas d'échec.
        """
        try:
            # Appliquer le rate limiting avant la requête
            self._rate_limit_request()
            # Utiliser OpenBB pour charger le cours historique sur la date donnée
            df_price = self.openbb.stocks.load(symbol=ticker, start_date=as_of_date, end_date=as_of_date)
            self.stats['total_requests'] += 1
            if df_price is None or df_price.empty:
                # Pas de donnée (jour non-tradé ou erreur)
                self.stats['failed_requests'] += 1
                return None
            # Récupérer le prix de clôture ajusté (Adj Close)
            price = df_price.iloc[-1]['Adj Close']
            self.stats['successful_requests'] += 1
            return float(price)
        except Exception as e:
            logging.error(f"Erreur lors de la récupération du prix sous-jacent: {e}")
            self.stats['failed_requests'] += 1
            return None
    
    def get_options_chain(self, ticker, as_of_date):
        """
        Récupère la chaîne d'options complète pour un ticker et une date donnée via OpenBB/Tradier.
        
        Args:
            ticker (str): Symbole du ticker.
            as_of_date (str): Date de fin de journée pour les données d'options (format 'YYYY-MM-DD').
        
        Returns:
            pd.DataFrame: DataFrame contenant tous les contrats d'options (avec leurs données) à cette date.
        """
        try:
            # Appliquer le rate limiting avant la requête
            self._rate_limit_request()
            # Récupérer la chaîne d'options à la date spécifiée (EOD) via OpenBB (source Tradier)
            chain = self.openbb.stocks.options.chains(symbol=ticker, source="Tradier", date=as_of_date)
            self.stats['total_requests'] += 1
            # Vérifier si des données sont retournées
            if chain is None or chain.empty:
                self.stats['failed_requests'] += 1
                return None
            # S'assurer que c'est un DataFrame pandas
            if not isinstance(chain, pd.DataFrame):
                # Si openbb renvoie un objet spécifique, essayer de le convertir
                try:
                    chain = chain.to_dataframe()
                except Exception:
                    chain = pd.DataFrame(chain)
            # Filtrer les contrats par date d'expiration (entre 7 et 730 jours)
            filtered_contracts = []
            as_of_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
            for _, opt in chain.iterrows():
                exp_str = opt['expiration'] if 'expiration' in opt else opt.get('expiration_date')
                if exp_str is None:
                    continue
                exp_date = pd.to_datetime(exp_str).date() if not isinstance(exp_str, date) else exp_str
                days_to_expiry = (exp_date - as_of_dt.date()).days
                if self.min_days_to_expiry <= days_to_expiry <= self.max_days_to_expiry:
                    filtered_contracts.append(opt)
            if not filtered_contracts:
                self.stats['successful_requests'] += 1  # Requête réussie mais aucun contrat dans le critère
                return pd.DataFrame()  # retourne un DataFrame vide si rien après filtrage
            df_filtered = pd.DataFrame(filtered_contracts)
            self.stats['successful_requests'] += 1
            return df_filtered.reset_index(drop=True)
        except Exception as e:
            logging.error(f"Erreur lors de la récupération de la chaîne d'options: {e}")
            self.stats['failed_requests'] += 1
            return None
    
    def calculate_greeks(self, option_type, S, K, T, r, sigma):
        """
        Calcule les Greeks (delta, gamma, theta, vega) selon le modèle de Black-Scholes.
        
        Args:
            option_type (str): 'call' ou 'put' (type d'option).
            S (float): Prix du sous-jacent.
            K (float): Strike (prix d'exercice de l'option).
            T (float): Temps jusqu'à expiration en années.
            r (float): Taux sans risque (décimal).
            sigma (float): Volatilité implicite (décimal).
        
        Returns:
            dict: Un dictionnaire contenant delta, gamma, theta, vega.
        """
        # Protection contre les valeurs non valides
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}
        # Calcul des facteurs d1 et d2 de Black-Scholes
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        # Calcul de Delta
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        # Calcul de Gamma
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        # Calcul de Theta (valeur par jour)
        term1 = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        if option_type.lower() == 'call':
            term2 = - r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        theta = (term1 + term2) / 365.0  # Theta par jour
        # Calcul de Vega
        vega = (S * norm.pdf(d1) * np.sqrt(T)) / 100.0  # vega exprimé par 1% de vol
        # Arrondir les résultats pour plus de lisibilité
        return {
            'delta': round(delta, 4),
            'gamma': round(gamma, 4),
            'theta': round(theta, 4),
            'vega': round(vega, 4)
        }
    
    def collect_daily_options_data(self, ticker, collection_date, risk_free_rate=0.05):
        """
        Collecte toutes les données d'options pour un ticker à une date donnée.
        
        Args:
            ticker (str): Symbole du sous-jacent.
            collection_date (str): Date de collecte (format 'YYYY-MM-DD').
            risk_free_rate (float): Taux sans risque à utiliser pour le calcul des Greeks.
        
        Returns:
            pd.DataFrame: DataFrame des options collectées pour cette date.
        """
        logging.info(f"Collecte des données d'options pour {ticker} au {collection_date}")
        
        # 1. Récupérer le prix du sous-jacent ce jour-là
        underlying_price = self.get_underlying_price(ticker, collection_date)
        if underlying_price is None:
            logging.warning(f"Prix sous-jacent indisponible pour {ticker} le {collection_date}")
            return pd.DataFrame()  # retourne vide si on ne peut pas récupérer le prix du sous-jacent
        
        # 2. Récupérer tous les contrats d'options (avec données) pour la date donnée
        contracts_df = self.get_options_chain(ticker, collection_date)
        if contracts_df is None:
            logging.warning(f"Aucune donnée d'options récupérée pour {ticker} le {collection_date}")
            return pd.DataFrame()
        if contracts_df.empty:
            # Chaîne récupérée mais aucune option dans la fourchette de maturité
            logging.info(f"Pas d'options éligibles (7-730 jours) pour {ticker} le {collection_date}")
            return pd.DataFrame()
        
        options_data_records = []
        
        # 3. Traiter chaque contrat d'option
        total_contracts = len(contracts_df)
        for i, (_, opt) in enumerate(contracts_df.iterrows()):
            if i % 50 == 0:
                logging.info(f"Traitement des contrats: {i}/{total_contracts} traités...")
            try:
                # Extraire les informations clés du contrat
                option_symbol = opt.get('symbol') or opt.get('optionSymbol')  # identifiant du contrat
                strike = float(opt['strike']) if 'strike' in opt else float(opt['strike_price'])
                # Obtenir la date d'expiration du contrat
                exp_date = opt['expiration'] if 'expiration' in opt else opt.get('expiration_date')
                exp_date = str(pd.to_datetime(exp_date).date())  # format 'YYYY-MM-DD'
                opt_type = opt['option_type'] if 'option_type' in opt else opt.get('optionType') or opt.get('type')
                if opt_type is None:
                    opt_type = "call" if option_symbol and "C" in option_symbol else "put"
                opt_type = opt_type.lower()
                
                # Calcul du temps jusqu'à expiration
                expiry_datetime = datetime.strptime(exp_date, "%Y-%m-%d")
                coll_datetime = datetime.strptime(collection_date, "%Y-%m-%d")
                days_to_expiry = (expiry_datetime - coll_datetime).days
                years_to_expiry = days_to_expiry / 365.25
                
                # Extraire le prix de clôture de l'option et le volume
                # On tente plusieurs champs possibles selon la source
                close_price = None
                if 'last' in opt and pd.notna(opt['last']):
                    close_price = opt['last']
                elif 'close' in opt and pd.notna(opt['close']):
                    close_price = opt['close']
                elif 'lastPrice' in opt:
                    close_price = opt['lastPrice']
                else:
                    close_price = opt.get('price', np.nan)
                close_price = 0.0 if close_price is None or pd.isna(close_price) else float(close_price)
                
                volume = 0
                if 'volume' in opt:
                    volume = int(opt['volume']) if not np.isnan(opt['volume']) else 0
                elif 'last_volume' in opt:
                    volume = int(opt['last_volume'])
                
                # Volatilité implicite (IV) - si fournie par la source, sinon calcul approximatif
                impl_vol = None
                # Tradier/ORATS: champ possible 'smv_vol' (vol implicite), ou 'impliedVolatility'
                if 'smv_vol' in opt and pd.notna(opt['smv_vol']):
                    impl_vol = float(opt['smv_vol'])
                elif 'impliedVolatility' in opt:
                    # Certains providers peuvent nommer 'impliedVolatility' (ex: YahooFinance)
                    impl_vol = float(opt['impliedVolatility'])
                elif 'mid_iv' in opt and pd.notna(opt['mid_iv']) and opt['mid_iv'] > 0:
                    impl_vol = float(opt['mid_iv'])
                # Si non fourni, estimer grossièrement la volatilité implicite
                if impl_vol is None or impl_vol <= 0:
                    moneyness = strike / underlying_price
                    base_vol = 0.20  # hypothèse de volatilité de base
                    smile_effect = 0.1 * abs(np.log(moneyness)) ** 1.5  # ajustement par rapport au moneyness
                    time_factor = np.sqrt(30.0 / max(days_to_expiry, 1))
                    impl_vol = (base_vol + smile_effect) * time_factor
                # S'assurer que sigma est bien en décimal (pas en %)
                if impl_vol > 1:  # s'il est probablement en pourcentage (ex: 25 = 25%)
                    impl_vol = impl_vol / 100.0
                
                # Calculer les Greeks via Black-Scholes
                greeks = self.calculate_greeks(
                    option_type=opt_type,
                    S=underlying_price,
                    K=strike,
                    T=years_to_expiry,
                    r=risk_free_rate,
                    sigma=impl_vol
                )
                
                # Déterminer si l'option est dans la monnaie (in the money)
                in_the_money = (opt_type == 'call' and underlying_price > strike) or \
                               (opt_type == 'put' and underlying_price < strike)
                
                # Créer l'enregistrement de la ligne d'option
                record = {
                    'dataDate': collection_date,
                    'ticker': ticker,
                    'optionSymbol': option_symbol if option_symbol else "",
                    'optionType': opt_type.upper(),
                    'strike': strike,
                    'expirationDate': exp_date,
                    'daysToExpiration': days_to_expiry,
                    'yearsToExpiration': round(years_to_expiry, 4),
                    'lastPrice': round(close_price, 4),
                    'volume': volume,
                    'impliedVolatility': round(impl_vol, 6),
                    'impliedVolatilityPct': round(impl_vol * 100, 2),
                    'underlyingPrice': round(underlying_price, 4),
                    'moneyness': round(strike / underlying_price, 4),
                    'delta': greeks['delta'],
                    'gamma': greeks['gamma'],
                    'theta': greeks['theta'],
                    'vega': greeks['vega'],
                    'inTheMoney': in_the_money
                }
                options_data_records.append(record)
                # Incrémenter le compteur d'options collectées
                self.stats['options_collected'] += 1
            except Exception as e:
                logging.error(f"Erreur lors du traitement d'un contrat: {e}")
                continue
        
        # Conversion de la liste de records en DataFrame
        df_options = pd.DataFrame(options_data_records)
        logging.info(f"{len(df_options)} options collectées pour {ticker} le {collection_date}")
        return df_options
    
    def collect_historical_data(self, ticker, days_back=180, end_date=None, parallel_days=5, save_intermediate=True):
        """
        Collecte les données d'options historiques pour un ticker sur plusieurs jours.
        
        Args:
            ticker (str): Symbole du ticker.
            days_back (int): Nombre de jours de données à récupérer (par défaut 180 jours ouvrés ~ 6 mois).
            end_date (str): Date de fin de la période (format 'YYYY-MM-DD'). Par défaut aujourd'hui.
            parallel_days (int): Nombre de jours à traiter en parallèle (threads).
            save_intermediate (bool): Si True, sauvegarde un CSV par jour collecté (données intermédiaires).
        
        Returns:
            pd.DataFrame: DataFrame combiné contenant toutes les données d'options collectées.
        """
        logging.info(f"Début de la collecte historique pour {ticker} sur {days_back} jours de bourse")
        
        # Déterminer la date de fin et la date de début de la collecte
        if end_date is None:
            end_date_dt = datetime.now().date()
        else:
            end_date_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        start_date_dt = end_date_dt - timedelta(days=days_back)
        
        # Générer la liste des dates de trading (exclusion des week-ends)
        trading_days = []
        current_date = start_date_dt
        # Utiliser le calendrier civil en supposant que days_back représente des jours civils,
        # puis filtrer pour ne garder que les jours de semaine.
        while current_date <= end_date_dt:
            if current_date.weekday() < 5:  # 0=lundi, ..., 4=vendredi
                trading_days.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
        logging.info(f"{len(trading_days)} jours de trading à collecter (week-ends exclus).")
        
        all_data_frames = []
        # Créer le répertoire de sortie pour les sauvegardes journalières
        output_dir = f"openbb_tradier_options_data/{ticker}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Traiter les jours en parallèle (par batch de 'parallel_days')
        with ThreadPoolExecutor(max_workers=parallel_days) as executor:
            for i in range(0, len(trading_days), parallel_days):
                batch = trading_days[i:i + parallel_days]
                futures = {}
                for day in batch:
                    futures[day] = executor.submit(self.collect_daily_options_data, ticker, day)
                # Attendre la fin de toutes les tâches du batch
                for day, future in futures.items():
                    try:
                        df_day = future.result(timeout=300)  # timeout de 5 minutes par jour
                        if df_day is not None and not df_day.empty:
                            all_data_frames.append(df_day)
                            # Sauvegarde intermédiaire si demandé
                            if save_intermediate:
                                day_file = f"{output_dir}/{ticker}_options_{day}.csv"
                                df_day.to_csv(day_file, index=False)
                                logging.info(f"Données journalières sauvegardées: {day_file}")
                        else:
                            logging.info(f"Aucune donnée collectée pour {day}")
                    except Exception as e:
                        logging.error(f"Erreur lors de la collecte du {day}: {e}")
                # Petite pause entre les lots de jours pour ne pas surcharger l'API
                time.sleep(2)
                # Afficher périodiquement les statistiques de collecte
                if (i // parallel_days + 1) % 4 == 0:  # toutes les 4 itérations de batch (~toutes les 20 jours si parallel_days=5)
                    self._print_statistics()
        
        # Combiner toutes les données collectées
        if all_data_frames:
            df_all = pd.concat(all_data_frames, ignore_index=True)
            # Trier les données combinées par date, type, strike, expiration
            df_all.sort_values(by=['dataDate', 'optionType', 'strike', 'expirationDate'], inplace=True)
            # Sauvegarder le CSV final combiné
            final_file = f"{output_dir}/{ticker}_options_historical_{days_back}d.csv"
            df_all.to_csv(final_file, index=False)
            logging.info(f"✅ Collecte historique terminée pour {ticker}: {len(df_all)} options collectées au total.")
            logging.info(f"📁 Données combinées sauvegardées dans: {final_file}")
            # Afficher les stats finales
            self._print_statistics()
            return df_all
        else:
            logging.warning("Aucune donnée d'option collectée sur la période demandée.")
            return pd.DataFrame()
    
    def _print_statistics(self):
        """
        Affiche les statistiques courantes de la collecte dans le log.
        """
        logging.info("📊 Statistiques de collecte :")
        logging.info(f"   - Requêtes totales envoyées : {self.stats['total_requests']}")
        logging.info(f"   - Requêtes réussies : {self.stats['successful_requests']}")
        logging.info(f"   - Requêtes échouées : {self.stats['failed_requests']}")
        logging.info(f"   - Options collectées : {self.stats['options_collected']}")
        if self.stats['total_requests'] > 0:
            success_rate = (self.stats['successful_requests'] / self.stats['total_requests']) * 100.0
            logging.info(f"   - Taux de succès des requêtes : {success_rate:.1f}%")
    
    def validate_data_quality(self, df):
        """
        Valide et nettoie le DataFrame d'options collectées selon des critères de qualité.
        
        Filtre les données aberrantes (prix trop bas, IV extrêmes) et ajoute des colonnes utiles.
        
        Args:
            df (pd.DataFrame): DataFrame des données d'options collectées.
        
        Returns:
            pd.DataFrame: DataFrame nettoyé et enrichi.
        """
        initial_count = len(df)
        # Filtrer les enregistrements selon des critères de qualité
        df = df[df['lastPrice'] > 0.01]            # prix last > 0.01
        df = df[df['impliedVolatilityPct'] > 1]    # IV > 1%
        df = df[df['impliedVolatilityPct'] < 500]  # IV < 500%
        df = df[df['volume'] >= 0]                 # volume non négatif
        df = df[df['yearsToExpiration'] > 0]       # options pas encore expirées à la date de collecte
        # Calcul des métriques de nettoyage
        removed_count = initial_count - len(df)
        removal_rate = (removed_count / initial_count * 100) if initial_count > 0 else 0.0
        logging.info("🧹 Nettoyage des données terminé :")
        logging.info(f"   - Enregistrements initiaux : {initial_count}")
        logging.info(f"   - Enregistrements supprimés : {removed_count} ({removal_rate:.1f}%)")
        logging.info(f"   - Enregistrements restants : {len(df)}")
        # Ajouter des colonnes supplémentaires utiles pour l'analyse
        df['spreadPct'] = ((df['strike'] - df['underlyingPrice']) / df['underlyingPrice']) * 100.0
        df['daysToExpiration'] = df['daysToExpiration'].astype(int)
        return df

# Fonction principale d'exemple d'utilisation
def main():
    """
    Exemple d'utilisation du collecteur OpenBB Tradier pour récupérer et analyser les options.
    """
    # Configuration utilisateur
    TRADIER_TOKEN = "VOTRE_CLE_API_TRADIER"  # Remplacez par votre clé d'API Tradier
    TICKER = "AAPL"
    DAYS_BACK = 180  # 6 mois d'historique environ
    
    print("🚀 Lancement du collecteur d'options (OpenBB Tradier)...")
    print("=" * 80)
    # Initialiser le collecteur avec la clé API Tradier
    collector = OpenBBTradierOptionsCollector(tradier_token=TRADIER_TOKEN)
    # Lancer la collecte historique
    df_history = collector.collect_historical_data(ticker=TICKER, days_back=DAYS_BACK, parallel_days=3, save_intermediate=True)
    if not df_history.empty:
        # Valider et nettoyer les données collectées
        df_clean = collector.validate_data_quality(df_history)
        # Afficher un résumé des données collectées
        print("\n📊 Résumé des données collectées :")
        print(f"   - Période couverte : du {df_clean['dataDate'].min()} au {df_clean['dataDate'].max()}")
        print(f"   - Nombre total d'options (après nettoyage) : {len(df_clean)}")
        print(f"   - Nombre de dates de collecte uniques : {df_clean['dataDate'].nunique()}")
        print(f"   - Nombre de dates d'expiration uniques : {df_clean['expirationDate'].nunique()}")
        print(f"   - Nombre de strikes uniques : {df_clean['strike'].nunique()}")
        # Répartition par type d'option
        print("\n📈 Répartition par type d'option :")
        print(df_clean['optionType'].value_counts())
        # Indication du fichier de sortie principal
        print(f"\n✅ Données d'options prêtes à l'utilisation. Fichier CSV consolidé enregistré dans 'openbb_tradier_options_data/{TICKER}/{TICKER}_options_historical_{DAYS_BACK}d.csv'")

# Exécuter l'exemple principal si ce script est exécuté directement
if __name__ == "__main__":
    main()

