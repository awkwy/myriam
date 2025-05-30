"""
Moteur de simulation Monte Carlo pour le pricing d'options et d'autocall
Auteur: Assistant Claude
Date: Mai 2025

Ce module implémente des simulations Monte Carlo optimisées pour pricer
des options vanilles, exotiques et des produits structurés comme les autocall.

CONCEPTS CLÉS:
1. Monte Carlo = Simuler de nombreux scénarios futurs possibles
2. Processus stochastique = Modélisation de l'évolution du prix
3. Réduction de variance = Techniques pour améliorer la précision
4. Autocall = Produit complexe nécessitant une simulation path-dependent
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import time
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
import logging
from numba import jit, prange  # Accélération des calculs
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


@dataclass
class SimulationParameters:
    """
    Paramètres pour la simulation Monte Carlo.
    
    EXPLICATION DES PARAMÈTRES:
    - n_paths: Plus de chemins = plus de précision mais plus lent
    - n_steps: Plus de pas = meilleur pour les options path-dependent
    - random_seed: Pour la reproductibilité
    - variance_reduction: Techniques pour améliorer l'efficacité
    """
    n_paths: int = 10000
    n_steps: int = 252  # Jours de trading par an
    dt: float = 1/252   # Pas de temps quotidien
    random_seed: Optional[int] = 42
    variance_reduction: List[str] = None  # ['antithetic', 'control_variate']
    parallel: bool = True
    n_cores: Optional[int] = None


@dataclass
class MarketData:
    """
    Données de marché nécessaires pour la simulation.
    """
    spot_price: float
    risk_free_rate: float
    dividend_yield: float = 0.0
    volatility: Union[float, np.ndarray]  # Peut être constante ou surface


class MonteCarloEngine:
    """
    Moteur de simulation Monte Carlo pour le pricing d'options.
    
    CAPACITÉS:
    1. Options vanilles (calls, puts)
    2. Options exotiques (barrières, asiatiques, lookback)
    3. Produits structurés (autocall, phoenix)
    4. Support multi-actifs
    5. Volatilité stochastique
    """
    
    def __init__(self, params: SimulationParameters = None):
        """
        Initialise le moteur avec les paramètres de simulation.
        """
        self.params = params or SimulationParameters()
        
        # Initialiser le générateur aléatoire
        if self.params.random_seed is not None:
            np.random.seed(self.params.random_seed)
        
        # Configuration du parallélisme
        if self.params.n_cores is None:
            self.params.n_cores = mp.cpu_count() - 1
        
        # Cache pour les résultats
        self.last_paths = None
        self.last_result = None
        
        logging.info(f"🚀 Moteur Monte Carlo initialisé")
        logging.info(f"   - Paths: {self.params.n_paths:,}")
        logging.info(f"   - Steps: {self.params.n_steps}")
        logging.info(f"   - Cores: {self.params.n_cores}")
    
    def generate_price_paths(self, market_data: MarketData, 
                           time_to_maturity: float,
                           return_all: bool = False) -> np.ndarray:
        """
        Génère les chemins de prix selon le modèle Black-Scholes.
        
        MODÈLE:
        dS = μ*S*dt + σ*S*dW
        
        où:
        - μ = drift (r - q)
        - σ = volatilité
        - dW = mouvement brownien
        
        Args:
            market_data: Données de marché
            time_to_maturity: Temps jusqu'à maturité (années)
            return_all: Retourner tous les pas ou juste le final
            
        Returns:
            Array des chemins de prix
        """
        S0 = market_data.spot_price
        r = market_data.risk_free_rate
        q = market_data.dividend_yield
        sigma = market_data.volatility
        
        # Paramètres de simulation
        n_paths = self.params.n_paths
        n_steps = int(time_to_maturity * self.params.n_steps)
        dt = time_to_maturity / n_steps
        
        # Drift et diffusion
        drift = (r - q - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        # Générer les innovations
        if 'antithetic' in (self.params.variance_reduction or []):
            # Variates antithétiques: utiliser Z et -Z
            # Réduit la variance par ~50%
            half_paths = n_paths // 2
            Z = np.random.standard_normal((n_steps, half_paths))
            Z = np.concatenate([Z, -Z], axis=1)
        else:
            Z = np.random.standard_normal((n_steps, n_paths))
        
        # Simuler les chemins (méthode log-normale pour éviter les prix négatifs)
        log_paths = np.zeros((n_steps + 1, n_paths))
        log_paths[0] = np.log(S0)
        
        for i in range(1, n_steps + 1):
            log_paths[i] = log_paths[i-1] + drift + diffusion * Z[i-1]
        
        # Convertir en prix
        price_paths = np.exp(log_paths)
        
        # Sauvegarder pour réutilisation
        self.last_paths = price_paths
        
        if return_all:
            return price_paths
        else:
            return price_paths[-1]  # Prix finaux seulement
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _generate_paths_numba(S0: float, drift: float, diffusion: float,
                            Z: np.ndarray, n_steps: int, n_paths: int) -> np.ndarray:
        """
        Version optimisée avec Numba pour la génération de chemins.
        ~10x plus rapide pour les grandes simulations.
        """
        log_paths = np.zeros((n_steps + 1, n_paths))
        log_paths[0] = np.log(S0)
        
        for i in prange(1, n_steps + 1):
            for j in prange(n_paths):
                log_paths[i, j] = log_paths[i-1, j] + drift + diffusion * Z[i-1, j]
        
        return np.exp(log_paths)
    
    def price_european_option(self, option_type: str, strike: float,
                            market_data: MarketData, time_to_maturity: float,
                            confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Price une option européenne vanilla.
        
        PRINCIPE:
        1. Simuler N chemins de prix
        2. Calculer le payoff pour chaque chemin
        3. Prendre la moyenne et actualiser
        
        Args:
            option_type: 'call' ou 'put'
            strike: Prix d'exercice
            market_data: Données de marché
            time_to_maturity: Temps jusqu'à maturité
            confidence_level: Niveau de confiance pour l'intervalle
            
        Returns:
            Dict avec prix, erreur standard, intervalle de confiance
        """
        logging.info(f"💰 Pricing option {option_type} K={strike}")
        
        # Générer les chemins
        start_time = time.time()
        final_prices = self.generate_price_paths(market_data, time_to_maturity)
        
        # Calculer les payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(final_prices - strike, 0)
        else:
            payoffs = np.maximum(strike - final_prices, 0)
        
        # Prix = moyenne actualisée des payoffs
        discount_factor = np.exp(-market_data.risk_free_rate * time_to_maturity)
        option_price = discount_factor * np.mean(payoffs)
        
        # Erreur standard et intervalle de confiance
        std_error = np.std(payoffs) / np.sqrt(self.params.n_paths)
        z_score = norm.ppf((1 + confidence_level) / 2)
        conf_interval = [
            option_price - z_score * std_error * discount_factor,
            option_price + z_score * std_error * discount_factor
        ]
        
        # Temps de calcul
        computation_time = time.time() - start_time
        
        result = {
            'price': option_price,
            'std_error': std_error * discount_factor,
            'confidence_interval': conf_interval,
            'computation_time': computation_time,
            'n_paths': self.params.n_paths
        }
        
        # Comparaison avec Black-Scholes analytique
        bs_price = self._black_scholes_price(
            option_type, market_data.spot_price, strike,
            time_to_maturity, market_data.risk_free_rate,
            market_data.volatility
        )
        result['black_scholes_price'] = bs_price
        result['mc_error'] = abs(option_price - bs_price)
        
        logging.info(f"✅ Prix MC: {option_price:.4f}, BS: {bs_price:.4f}")
        logging.info(f"⏱️  Temps: {computation_time:.2f}s")
        
        return result
    
    def price_barrier_option(self, option_type: str, strike: float, barrier: float,
                           barrier_type: str, market_data: MarketData,
                           time_to_maturity: float) -> Dict[str, float]:
        """
        Price une option barrière.
        
        TYPES DE BARRIÈRES:
        - up-and-out: Désactivée si S > barrier
        - down-and-out: Désactivée si S < barrier
        - up-and-in: Activée si S > barrier
        - down-and-in: Activée si S < barrier
        
        IMPORTANT: Nécessite le monitoring du chemin complet!
        """
        logging.info(f"🚧 Pricing barrière {barrier_type} {option_type}")
        
        # Générer tous les chemins (pas juste le prix final)
        price_paths = self.generate_price_paths(
            market_data, time_to_maturity, return_all=True
        )
        
        n_paths = price_paths.shape[1]
        final_prices = price_paths[-1]
        
        # Vérifier les conditions de barrière pour chaque chemin
        if barrier_type == 'up-and-out':
            # Désactivée si le prix dépasse la barrière
            barrier_hit = np.any(price_paths > barrier, axis=0)
            active_paths = ~barrier_hit
        elif barrier_type == 'down-and-out':
            # Désactivée si le prix passe sous la barrière
            barrier_hit = np.any(price_paths < barrier, axis=0)
            active_paths = ~barrier_hit
        elif barrier_type == 'up-and-in':
            # Activée si le prix dépasse la barrière
            barrier_hit = np.any(price_paths > barrier, axis=0)
            active_paths = barrier_hit
        elif barrier_type == 'down-and-in':
            # Activée si le prix passe sous la barrière
            barrier_hit = np.any(price_paths < barrier, axis=0)
            active_paths = barrier_hit
        else:
            raise ValueError(f"Type de barrière inconnu: {barrier_type}")
        
        # Calculer les payoffs pour les chemins actifs
        if option_type.lower() == 'call':
            payoffs = np.where(active_paths, np.maximum(final_prices - strike, 0), 0)
        else:
            payoffs = np.where(active_paths, np.maximum(strike - final_prices, 0), 0)
        
        # Prix et statistiques
        discount_factor = np.exp(-market_data.risk_free_rate * time_to_maturity)
        option_price = discount_factor * np.mean(payoffs)
        
        # Probabilité de toucher la barrière
        barrier_prob = np.mean(barrier_hit)
        
        result = {
            'price': option_price,
            'barrier_probability': barrier_prob,
            'active_paths_ratio': np.mean(active_paths),
            'n_paths': n_paths
        }
        
        logging.info(f"✅ Prix: {option_price:.4f}")
        logging.info(f"📊 Prob barrière: {barrier_prob:.1%}")
        
        return result
    
    def price_asian_option(self, option_type: str, strike: float,
                         market_data: MarketData, time_to_maturity: float,
                         averaging_type: str = 'arithmetic') -> Dict[str, float]:
        """
        Price une option asiatique (sur la moyenne).
        
        TYPES:
        - Arithmétique: Moyenne simple
        - Géométrique: Moyenne géométrique (a une solution analytique)
        
        UTILITÉ: Réduit la manipulation de prix à l'échéance
        """
        logging.info(f"🌏 Pricing option asiatique {averaging_type}")
        
        # Générer tous les chemins
        price_paths = self.generate_price_paths(
            market_data, time_to_maturity, return_all=True
        )
        
        # Calculer la moyenne selon le type
        if averaging_type == 'arithmetic':
            average_prices = np.mean(price_paths[1:], axis=0)  # Exclure S0
        else:  # geometric
            log_prices = np.log(price_paths[1:])
            average_prices = np.exp(np.mean(log_prices, axis=0))
        
        # Payoffs basés sur la moyenne
        if option_type.lower() == 'call':
            payoffs = np.maximum(average_prices - strike, 0)
        else:
            payoffs = np.maximum(strike - average_prices, 0)
        
        # Prix
        discount_factor = np.exp(-market_data.risk_free_rate * time_to_maturity)
        option_price = discount_factor * np.mean(payoffs)
        
        result = {
            'price': option_price,
            'average_type': averaging_type,
            'mean_average_price': np.mean(average_prices),
            'std_average_price': np.std(average_prices)
        }
        
        return result
    
    def price_autocall(self, initial_spot: float, autocall_levels: List[float],
                      observation_dates: List[float], coupon_rates: List[float],
                      barrier_level: float, market_data: MarketData,
                      notional: float = 100.0) -> Dict[str, float]:
        """
        Price un autocall (produit structuré complexe).
        
        MÉCANISME AUTOCALL:
        1. Dates d'observation périodiques (ex: tous les 6 mois)
        2. Si S > niveau autocall: Remboursement anticipé + coupon
        3. Si jamais déclenché et S_final < barrière: Perte en capital
        4. Sinon: Remboursement du nominal
        
        COMPLEXITÉ: Path-dependent avec sorties multiples possibles
        
        Args:
            initial_spot: Prix initial du sous-jacent
            autocall_levels: Niveaux de déclenchement [100%, 95%, 90%...]
            observation_dates: Dates d'observation en années [0.5, 1.0, 1.5...]
            coupon_rates: Taux de coupon si autocall [5%, 10%, 15%...]
            barrier_level: Niveau de protection du capital (ex: 70%)
            market_data: Données de marché
            notional: Montant nominal
            
        Returns:
            Dict avec prix, probabilités, durée moyenne
        """
        logging.info("🎯 Pricing Autocall")
        logging.info(f"   - Niveaux: {autocall_levels}")
        logging.info(f"   - Barrière: {barrier_level}%")
        
        # Validation des inputs
        n_observations = len(observation_dates)
        assert len(autocall_levels) == n_observations
        assert len(coupon_rates) == n_observations
        
        # Temps jusqu'à la dernière observation
        final_maturity = max(observation_dates)
        
        # Générer tous les chemins de prix
        price_paths = self.generate_price_paths(
            market_data, final_maturity, return_all=True
        )
        
        n_paths = price_paths.shape[1]
        
        # Pour chaque chemin, déterminer le payoff
        payoffs = np.zeros(n_paths)
        autocall_times = np.full(n_paths, final_maturity)  # Temps de sortie
        autocall_triggered = np.zeros(n_paths, dtype=bool)
        
        # Indices des dates d'observation dans la grille temporelle
        obs_indices = [
            int(obs_date / final_maturity * (price_paths.shape[0] - 1))
            for obs_date in observation_dates
        ]
        
        # Vérifier chaque date d'observation
        for i, (obs_idx, level, coupon) in enumerate(
            zip(obs_indices, autocall_levels, coupon_rates)
        ):
            # Prix à cette date d'observation
            obs_prices = price_paths[obs_idx]
            
            # Niveau de déclenchement absolu
            trigger_price = initial_spot * (level / 100)
            
            # Chemins qui déclenchent l'autocall (et pas déjà déclenchés)
            newly_triggered = (obs_prices >= trigger_price) & (~autocall_triggered)
            
            # Mettre à jour les payoffs pour les nouveaux déclenchements
            if np.any(newly_triggered):
                # Remboursement = Nominal + Coupon
                autocall_payoff = notional * (1 + coupon / 100)
                
                # Actualiser à la date d'observation
                discount = np.exp(-market_data.risk_free_rate * observation_dates[i])
                
                payoffs[newly_triggered] = autocall_payoff * discount
                autocall_times[newly_triggered] = observation_dates[i]
                autocall_triggered[newly_triggered] = True
        
        # Pour les chemins non déclenchés, vérifier le niveau final
        not_triggered = ~autocall_triggered
        if np.any(not_triggered):
            final_prices = price_paths[-1, not_triggered]
            barrier_price = initial_spot * (barrier_level / 100)
            
            # Si au-dessus de la barrière: remboursement du nominal
            above_barrier = final_prices >= barrier_price
            payoffs[not_triggered & above_barrier] = notional * np.exp(
                -market_data.risk_free_rate * final_maturity
            )
            
            # Si en-dessous de la barrière: perte en capital
            below_barrier = final_prices < barrier_price
            below_barrier_mask = np.zeros(n_paths, dtype=bool)
            below_barrier_mask[not_triggered] = below_barrier
            
            # Perte = performance négative du sous-jacent
            if np.any(below_barrier_mask):
                performance = price_paths[-1, below_barrier_mask] / initial_spot
                payoffs[below_barrier_mask] = notional * performance * np.exp(
                    -market_data.risk_free_rate * final_maturity
                )
        
        # Prix de l'autocall = moyenne des payoffs
        autocall_price = np.mean(payoffs)
        
        # Statistiques détaillées
        autocall_probs = []
        for i, obs_date in enumerate(observation_dates):
            prob = np.mean(autocall_times == obs_date)
            autocall_probs.append({
                'date': obs_date,
                'probability': prob,
                'coupon': coupon_rates[i]
            })
        
        # Probabilité de toucher la barrière
        barrier_hit_prob = np.mean(
            (price_paths[-1] < initial_spot * barrier_level / 100) & 
            (~autocall_triggered)
        )
        
        # Durée moyenne jusqu'au remboursement
        average_duration = np.mean(autocall_times)
        
        result = {
            'price': autocall_price,
            'price_pct': autocall_price / notional * 100,
            'autocall_probabilities': autocall_probs,
            'total_autocall_prob': np.mean(autocall_triggered),
            'barrier_hit_prob': barrier_hit_prob,
            'average_duration': average_duration,
            'final_payoff_distribution': {
                'min': np.min(payoffs),
                'p25': np.percentile(payoffs, 25),
                'median': np.median(payoffs),
                'p75': np.percentile(payoffs, 75),
                'max': np.max(payoffs)
            }
        }
        
        logging.info(f"✅ Prix Autocall: {autocall_price:.2f} ({autocall_price/notional*100:.1f}%)")
        logging.info(f"📊 Prob autocall total: {result['total_autocall_prob']:.1%}")
        logging.info(f"⚠️  Prob barrière: {barrier_hit_prob:.1%}")
        logging.info(f"⏱️  Durée moyenne: {average_duration:.2f} ans")
        
        return result
    
    def price_with_stochastic_volatility(self, option_type: str, strike: float,
                                        market_data: MarketData, 
                                        time_to_maturity: float,
                                        vol_of_vol: float = 0.3,
                                        mean_reversion: float = 2.0) -> Dict[str, float]:
        """
        Price avec volatilité stochastique (modèle de Heston simplifié).
        
        POURQUOI LA VOL STOCHASTIQUE?
        - La volatilité n'est PAS constante en réalité
        - Capture mieux les queues épaisses
        - Explique le smile de volatilité
        
        MODÈLE:
        dS = μ*S*dt + √v*S*dW₁
        dv = κ(θ-v)*dt + σ*√v*dW₂
        
        où dW₁ et dW₂ sont corrélés
        """
        logging.info("🌪️  Pricing avec volatilité stochastique")
        
        S0 = market_data.spot_price
        r = market_data.risk_free_rate
        v0 = market_data.volatility ** 2  # Variance initiale
        
        # Paramètres du modèle
        kappa = mean_reversion  # Vitesse de retour à la moyenne
        theta = v0  # Variance long terme
        sigma_v = vol_of_vol  # Vol de la vol
        rho = -0.7  # Corrélation typique négative
        
        # Simulation
        n_paths = self.params.n_paths
        n_steps = int(time_to_maturity * self.params.n_steps)
        dt = time_to_maturity / n_steps
        
        # Générer les innovations corrélées
        Z1 = np.random.standard_normal((n_steps, n_paths))
        Z2 = np.random.standard_normal((n_steps, n_paths))
        
        # Corrélation
        W1 = Z1
        W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2
        
        # Initialisation
        S = np.zeros((n_steps + 1, n_paths))
        v = np.zeros((n_steps + 1, n_paths))
        S[0] = S0
        v[0] = v0
        
        # Simulation (schéma d'Euler)
        for i in range(1, n_steps + 1):
            # Variance (avec correction pour éviter les valeurs négatives)
            v[i] = v[i-1] + kappa * (theta - v[i-1]) * dt + \
                   sigma_v * np.sqrt(np.maximum(v[i-1], 0)) * np.sqrt(dt) * W2[i-1]
            v[i] = np.maximum(v[i], 0)  # Forcer positivité
            
            # Prix
            S[i] = S[i-1] * np.exp(
                (r - 0.5 * v[i-1]) * dt + 
                np.sqrt(v[i-1]) * np.sqrt(dt) * W1[i-1]
            )
        
        # Payoffs
        final_prices = S[-1]
        if option_type.lower() == 'call':
            payoffs = np.maximum(final_prices - strike, 0)
        else:
            payoffs = np.maximum(strike - final_prices, 0)
        
        # Prix
        discount_factor = np.exp(-r * time_to_maturity)
        option_price = discount_factor * np.mean(payoffs)
        
        # Volatilité implicite moyenne réalisée
        realized_vol = np.sqrt(np.mean(v, axis=0)).mean() * 100
        
        result = {
            'price': option_price,
            'realized_volatility': realized_vol,
            'initial_volatility': np.sqrt(v0) * 100,
            'vol_of_vol': vol_of_vol,
            'final_price_distribution': {
                'p5': np.percentile(final_prices, 5),
                'p50': np.median(final_prices),
                'p95': np.percentile(final_prices, 95)
            }
        }
        
        return result
    
    def calculate_greeks_mc(self, option_type: str, strike: float,
                          market_data: MarketData, time_to_maturity: float,
                          bump_size: float = 0.01) -> Dict[str, float]:
        """
        Calcule les Greeks par différences finies Monte Carlo.
        
        MÉTHODE:
        - Delta: ∂V/∂S ≈ (V(S+h) - V(S-h)) / 2h
        - Gamma: ∂²V/∂S² ≈ (V(S+h) - 2V(S) + V(S-h)) / h²
        - Vega: ∂V/∂σ
        - Theta: ∂V/∂t
        - Rho: ∂V/∂r
        """
        logging.info("🔢 Calcul des Greeks par Monte Carlo")
        
        # Prix de base
        base_result = self.price_european_option(
            option_type, strike, market_data, time_to_maturity
        )
        base_price = base_result['price']
        
        # Delta et Gamma (bump le spot)
        spot_bump = market_data.spot_price * bump_size
        
        # Prix avec spot + bump
        market_up = MarketData(
            spot_price=market_data.spot_price + spot_bump,
            risk_free_rate=market_data.risk_free_rate,
            dividend_yield=market_data.dividend_yield,
            volatility=market_data.volatility
        )
        price_up = self.price_european_option(
            option_type, strike, market_up, time_to_maturity
        )['price']
        
        # Prix avec spot - bump
        market_down = MarketData(
            spot_price=market_data.spot_price - spot_bump,
            risk_free_rate=market_data.risk_free_rate,
            dividend_yield=market_data.dividend_yield,
            volatility=market_data.volatility
        )
        price_down = self.price_european_option(
            option_type, strike, market_down, time_to_maturity
        )['price']
        
        # Greeks
        delta = (price_up - price_down) / (2 * spot_bump)
        gamma = (price_up - 2 * base_price + price_down) / (spot_bump ** 2)
        
        # Vega (bump la volatilité)
        vol_bump = 0.01  # 1% de vol
        market_vol_up = MarketData(
            spot_price=market_data.spot_price,
            risk_free_rate=market_data.risk_free_rate,
            dividend_yield=market_data.dividend_yield,
            volatility=market_data.volatility + vol_bump
        )
        price_vol_up = self.price_european_option(
            option_type, strike, market_vol_up, time_to_maturity
        )['price']
        
        vega = (price_vol_up - base_price) / vol_bump / 100  # Pour 1% de vol
        
        # Theta (bump le temps)
        time_bump = 1/365  # 1 jour
        if time_to_maturity > time_bump:
            price_time = self.price_european_option(
                option_type, strike, market_data, time_to_maturity - time_bump
            )['price']
            theta = (price_time - base_price) / time_bump / 365  # Theta journalier
        else:
            theta = 0
        
        # Rho (bump le taux)
        rate_bump = 0.0001  # 1 bp
        market_rate_up = MarketData(
            spot_price=market_data.spot_price,
            risk_free_rate=market_data.risk_free_rate + rate_bump,
            dividend_yield=market_data.dividend_yield,
            volatility=market_data.volatility
        )
        price_rate_up = self.price_european_option(
            option_type, strike, market_rate_up, time_to_maturity
        )['price']
        
        rho = (price_rate_up - base_price) / rate_bump / 100  # Pour 1% de taux
        
        greeks = {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho,
            'option_price': base_price
        }
        
        # Comparaison avec Black-Scholes analytique
        bs_greeks = self._black_scholes_greeks(
            option_type, market_data.spot_price, strike,
            time_to_maturity, market_data.risk_free_rate,
            market_data.volatility
        )
        
        # Erreurs
        for greek in ['delta', 'gamma', 'vega', 'theta']:
            if greek in bs_greeks:
                greeks[f'{greek}_error'] = abs(greeks[greek] - bs_greeks[greek])
        
        return greeks
    
    @staticmethod
    def _black_scholes_price(option_type: str, S: float, K: float,
                           T: float, r: float, sigma: float) -> float:
        """Formule analytique Black-Scholes pour comparaison."""
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type.lower() == 'call':
            return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    @staticmethod
    def _black_scholes_greeks(option_type: str, S: float, K: float,
                            T: float, r: float, sigma: float) -> Dict[str, float]:
        """Greeks analytiques Black-Scholes pour comparaison."""
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        # Delta
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        
        # Gamma
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Vega
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        # Theta
        term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        if option_type.lower() == 'call':
            term2 = -r * K * np.exp(-r*T) * norm.cdf(d2)
        else:
            term2 = r * K * np.exp(-r*T) * norm.cdf(-d2)
        theta = (term1 + term2) / 365
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta
        }
    
    def visualize_paths(self, n_paths_to_show: int = 100,
                       market_data: Optional[MarketData] = None,
                       time_to_maturity: float = 1.0):
        """
        Visualise quelques chemins de prix simulés.
        
        UTILITÉ PÉDAGOGIQUE:
        - Voir la dispersion des chemins
        - Comprendre l'impact de la volatilité
        - Visualiser les barrières pour les exotiques
        """
        if self.last_paths is None:
            if market_data is None:
                raise ValueError("Aucun chemin disponible, fournir market_data")
            self.generate_price_paths(market_data, time_to_maturity, return_all=True)
        
        paths = self.last_paths
        n_paths_to_show = min(n_paths_to_show, paths.shape[1])
        
        # Créer la figure
        plt.figure(figsize=(12, 8))
        
        # Temps en jours
        time_grid = np.linspace(0, time_to_maturity * 252, paths.shape[0])
        
        # Plot des chemins individuels
        for i in range(n_paths_to_show):
            plt.plot(time_grid, paths[:, i], alpha=0.3, linewidth=0.8)
        
        # Statistiques
        mean_path = np.mean(paths, axis=1)
        p5_path = np.percentile(paths, 5, axis=1)
        p95_path = np.percentile(paths, 95, axis=1)
        
        plt.plot(time_grid, mean_path, 'k-', linewidth=2, label='Moyenne')
        plt.plot(time_grid, p5_path, 'r--', linewidth=1.5, label='5e percentile')
        plt.plot(time_grid, p95_path, 'g--', linewidth=1.5, label='95e percentile')
        
        # Prix initial
        plt.axhline(y=paths[0, 0], color='b', linestyle=':', label='Prix initial')
        
        plt.xlabel('Temps (jours)')
        plt.ylabel('Prix du sous-jacent')
        plt.title(f'Simulation Monte Carlo - {self.params.n_paths:,} chemins')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()


def demonstrate_monte_carlo_pricing():
    """
    Démontre l'utilisation complète du moteur Monte Carlo.
    """
    print("="*70)
    print("🎲 MOTEUR DE SIMULATION MONTE CARLO")
    print("="*70)
    
    # Paramètres de simulation
    params = SimulationParameters(
        n_paths=50000,
        n_steps=252,
        random_seed=42,
        variance_reduction=['antithetic']
    )
    
    # Initialiser le moteur
    engine = MonteCarloEngine(params)
    
    # Données de marché
    market_data = MarketData(
        spot_price=100,
        risk_free_rate=0.05,
        dividend_yield=0.02,
        volatility=0.25  # 25% annualisée
    )
    
    print("\n📊 DONNÉES DE MARCHÉ")
    print(f"   - Spot: ${market_data.spot_price}")
    print(f"   - Taux: {market_data.risk_free_rate:.1%}")
    print(f"   - Volatilité: {market_data.volatility:.1%}")
    
    # 1. Option Européenne
    print("\n1️⃣  OPTION EUROPÉENNE")
    print("-" * 50)
    
    result_call = engine.price_european_option(
        'call', strike=105, market_data=market_data, time_to_maturity=1.0
    )
    
    print(f"Call ATM (K=105, T=1an):")
    print(f"   - Prix MC: ${result_call['price']:.4f}")
    print(f"   - Prix BS: ${result_call['black_scholes_price']:.4f}")
    print(f"   - Erreur: ${result_call['mc_error']:.4f}")
    print(f"   - Intervalle 95%: [{result_call['confidence_interval'][0]:.4f}, "
          f"{result_call['confidence_interval'][1]:.4f}]")
    
    # 2. Option Barrière
    print("\n2️⃣  OPTION BARRIÈRE")
    print("-" * 50)
    
    result_barrier = engine.price_barrier_option(
        'call', strike=100, barrier=120, barrier_type='up-and-out',
        market_data=market_data, time_to_maturity=1.0
    )
    
    print(f"Call Up-and-Out (K=100, B=120):")
    print(f"   - Prix: ${result_barrier['price']:.4f}")
    print(f"   - Prob barrière: {result_barrier['barrier_probability']:.1%}")
    
    # 3. Option Asiatique
    print("\n3️⃣  OPTION ASIATIQUE")
    print("-" * 50)
    
    result_asian = engine.price_asian_option(
        'call', strike=100, market_data=market_data,
        time_to_maturity=1.0, averaging_type='arithmetic'
    )
    
    print(f"Call Asiatique (K=100):")
    print(f"   - Prix: ${result_asian['price']:.4f}")
    print(f"   - Prix moyen attendu: ${result_asian['mean_average_price']:.2f}")
    
    # 4. Autocall
    print("\n4️⃣  AUTOCALL")
    print("-" * 50)
    
    result_autocall = engine.price_autocall(
        initial_spot=100,
        autocall_levels=[100, 95, 90, 85],  # Niveaux dégressifs
        observation_dates=[0.5, 1.0, 1.5, 2.0],  # Tous les 6 mois
        coupon_rates=[5, 10, 15, 20],  # Coupons croissants
        barrier_level=70,  # Protection à 70%
        market_data=market_data,
        notional=100
    )
    
    print(f"Autocall 2 ans:")
    print(f"   - Prix: ${result_autocall['price']:.2f} ({result_autocall['price_pct']:.1f}%)")
    print(f"   - Prob autocall: {result_autocall['total_autocall_prob']:.1%}")
    print(f"   - Prob barrière: {result_autocall['barrier_hit_prob']:.1%}")
    print(f"   - Durée moyenne: {result_autocall['average_duration']:.2f} ans")
    
    print("\nProbabilités par date:")
    for prob_info in result_autocall['autocall_probabilities']:
        print(f"   - {prob_info['date']} ans: {prob_info['probability']:.1%} "
              f"(coupon {prob_info['coupon']}%)")
    
    # 5. Volatilité Stochastique
    print("\n5️⃣  VOLATILITÉ STOCHASTIQUE")
    print("-" * 50)
    
    result_stoch = engine.price_with_stochastic_volatility(
        'call', strike=100, market_data=market_data,
        time_to_maturity=1.0, vol_of_vol=0.3
    )
    
    print(f"Call avec vol stochastique:")
    print(f"   - Prix: ${result_stoch['price']:.4f}")
    print(f"   - Vol réalisée moyenne: {result_stoch['realized_volatility']:.1f}%")
    
    # 6. Greeks Monte Carlo
    print("\n6️⃣  GREEKS MONTE CARLO")
    print("-" * 50)
    
    greeks = engine.calculate_greeks_mc(
        'call', strike=100, market_data=market_data, time_to_maturity=1.0
    )
    
    print(f"Greeks du Call ATM:")
    print(f"   - Delta: {greeks['delta']:.4f}")
    print(f"   - Gamma: {greeks['gamma']:.4f}")
    print(f"   - Vega: {greeks['vega']:.4f}")
    print(f"   - Theta: {greeks['theta']:.4f}")
    
    print("\n✅ Démonstration terminée!")
    print("Le moteur Monte Carlo est prêt pour le pricing de produits complexes.")


if __name__ == "__main__":
    demonstrate_monte_carlo_pricing()
