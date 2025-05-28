"""
Constructeur et analyseur de surfaces de volatilité
Auteur: Assistant Claude
Date: Mai 2025

Ce module transforme les données d'options collectées en surfaces de volatilité
exploitables pour le pricing et l'analyse de risque.

CONCEPTS FONDAMENTAUX:
1. Surface de volatilité = représentation 3D de σ(K, T)
2. Interpolation = estimer σ pour des (K,T) non observés
3. Arbitrage-free = contraintes pour éviter les opportunités d'arbitrage
"""

import pandas as pd
import numpy as np
from scipy.interpolate import griddata, RBFInterpolator, interp2d
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import logging
from typing import Dict, List, Tuple, Optional
import json

warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class VolatilitySurfaceBuilder:
    """
    Construit et analyse des surfaces de volatilité à partir de données d'options.
    
    POURQUOI DES SURFACES DE VOLATILITÉ?
    - Le modèle Black-Scholes assume σ constant (faux en pratique!)
    - En réalité, σ varie avec K (smile) et T (term structure)
    - La surface capture ces variations pour un pricing précis
    """
    
    def __init__(self):
        self.surfaces = {}  # Stockage des surfaces par date
        self.surface_metrics = {}  # Métriques calculées
        self.interpolation_methods = {
            'linear': self._interpolate_linear,
            'cubic': self._interpolate_cubic,
            'rbf': self._interpolate_rbf,
            'svi': self._interpolate_svi  # Stochastic Volatility Inspired
        }
        
    def build_surface_from_data(self, options_df: pd.DataFrame, 
                               surface_date: str,
                               method: str = 'rbf') -> Dict:
        """
        Construit une surface de volatilité pour une date donnée.
        
        Args:
            options_df: DataFrame avec les données d'options
            surface_date: Date de la surface (format YYYY-MM-DD)
            method: Méthode d'interpolation
            
        Returns:
            Dict contenant la surface et les métriques
        """
        logging.info(f"🏗️  Construction surface pour {surface_date} avec méthode {method}")
        
        # Filtrer les données pour cette date
        daily_data = options_df[options_df['dataDate'] == surface_date].copy()
        
        if len(daily_data) < 10:
            logging.warning(f"⚠️  Pas assez de points ({len(daily_data)}) pour {surface_date}")
            return None
        
        # Extraire les coordonnées et valeurs
        strikes = daily_data['strike'].values
        ttms = daily_data['yearsToExpiration'].values
        vols = daily_data['impliedVolatility'].values
        spot = daily_data['spotPrice'].iloc[0]
        
        # Filtrer les valeurs aberrantes
        mask = self._filter_outliers(strikes, ttms, vols, spot)
        strikes_clean = strikes[mask]
        ttms_clean = ttms[mask]
        vols_clean = vols[mask]
        
        logging.info(f"📊 Points valides: {len(vols_clean)}/{len(vols)}")
        
        # Créer la grille d'interpolation
        strike_grid, ttm_grid = self._create_interpolation_grid(
            strikes_clean, ttms_clean, spot
        )
        
        # Interpoler la surface
        if method in self.interpolation_methods:
            vol_surface = self.interpolation_methods[method](
                strikes_clean, ttms_clean, vols_clean, 
                strike_grid, ttm_grid
            )
        else:
            raise ValueError(f"Méthode inconnue: {method}")
        
        # Calculer les métriques de la surface
        metrics = self._calculate_surface_metrics(
            daily_data, strikes_clean, ttms_clean, vols_clean, spot
        )
        
        # Vérifier l'absence d'arbitrage
        arbitrage_free = self._check_arbitrage_conditions(
            strike_grid, ttm_grid, vol_surface, spot
        )
        
        # Créer l'objet surface
        surface = {
            'date': surface_date,
            'spot_price': spot,
            'strike_grid': strike_grid,
            'ttm_grid': ttm_grid,
            'vol_surface': vol_surface,
            'raw_strikes': strikes_clean,
            'raw_ttms': ttms_clean,
            'raw_vols': vols_clean,
            'metrics': metrics,
            'arbitrage_free': arbitrage_free,
            'method': method,
            'n_points': len(vols_clean)
        }
        
        # Stocker la surface
        self.surfaces[surface_date] = surface
        self.surface_metrics[surface_date] = metrics
        
        return surface
    
    def _filter_outliers(self, strikes: np.ndarray, ttms: np.ndarray, 
                        vols: np.ndarray, spot: float) -> np.ndarray:
        """
        Filtre les points aberrants selon plusieurs critères.
        
        CRITÈRES:
        1. Volatilité réaliste (5% - 200%)
        2. Moneyness raisonnable (50% - 200%)
        3. TTM > 0
        4. Cohérence locale (pas de sauts brutaux)
        """
        # Critères de base
        vol_mask = (vols >= 5) & (vols <= 200)
        moneyness = strikes / spot
        moneyness_mask = (moneyness >= 0.5) & (moneyness <= 2.0)
        ttm_mask = ttms > 0.01  # Au moins quelques jours
        
        # Combiner les masques
        mask = vol_mask & moneyness_mask & ttm_mask
        
        # Filtrage statistique supplémentaire
        if mask.sum() > 10:
            # Enlever les outliers > 3 écarts-types
            vol_mean = vols[mask].mean()
            vol_std = vols[mask].std()
            statistical_mask = np.abs(vols - vol_mean) <= 3 * vol_std
            mask = mask & statistical_mask
        
        return mask
    
    def _create_interpolation_grid(self, strikes: np.ndarray, ttms: np.ndarray,
                                  spot: float, n_strikes: int = 50, 
                                  n_ttms: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crée une grille régulière pour l'interpolation.
        
        DESIGN:
        - Plus dense près d'ATM (plus important pour le trading)
        - Couvre 70% - 130% de moneyness typiquement
        - TTM de 1 semaine à 2 ans
        """
        # Grille en moneyness plutôt qu'en strikes absolus
        moneyness_min = max(0.7, (strikes / spot).min())
        moneyness_max = min(1.3, (strikes / spot).max())
        
        # Grille plus dense près d'ATM
        moneyness_grid = np.concatenate([
            np.linspace(moneyness_min, 0.95, n_strikes // 3),
            np.linspace(0.95, 1.05, n_strikes // 3),
            np.linspace(1.05, moneyness_max, n_strikes // 3)
        ])
        
        strike_grid = moneyness_grid * spot
        
        # Grille temporelle (plus dense court terme)
        ttm_min = max(7/365, ttms.min())  # Au moins 1 semaine
        ttm_max = min(2.0, ttms.max())    # Max 2 ans
        
        ttm_grid = np.concatenate([
            np.linspace(ttm_min, 0.25, n_ttms // 3),    # Court terme
            np.linspace(0.25, 1.0, n_ttms // 3),        # Moyen terme
            np.linspace(1.0, ttm_max, n_ttms // 3)      # Long terme
        ])
        
        # Créer le meshgrid
        strike_mesh, ttm_mesh = np.meshgrid(strike_grid, ttm_grid)
        
        return strike_mesh, ttm_mesh
    
    def _interpolate_linear(self, strikes: np.ndarray, ttms: np.ndarray, 
                           vols: np.ndarray, strike_grid: np.ndarray, 
                           ttm_grid: np.ndarray) -> np.ndarray:
        """Interpolation linéaire - rapide mais peu smooth"""
        points = np.column_stack((strikes, ttms))
        return griddata(points, vols, (strike_grid, ttm_grid), method='linear')
    
    def _interpolate_cubic(self, strikes: np.ndarray, ttms: np.ndarray, 
                          vols: np.ndarray, strike_grid: np.ndarray, 
                          ttm_grid: np.ndarray) -> np.ndarray:
        """Interpolation cubique - bon compromis smooth/précision"""
        points = np.column_stack((strikes, ttms))
        return griddata(points, vols, (strike_grid, ttm_grid), method='cubic')
    
    def _interpolate_rbf(self, strikes: np.ndarray, ttms: np.ndarray, 
                        vols: np.ndarray, strike_grid: np.ndarray, 
                        ttm_grid: np.ndarray) -> np.ndarray:
        """
        Interpolation RBF (Radial Basis Function) - très smooth.
        
        AVANTAGES:
        - Surface très lisse
        - Extrapolation raisonnable
        - Bon pour la visualisation
        
        INCONVÉNIENTS:
        - Plus lent
        - Peut sur-lisser
        """
        points = np.column_stack((strikes, ttms))
        rbf = RBFInterpolator(points, vols, kernel='thin_plate_spline', epsilon=2)
        
        # Reshape pour l'interpolation
        grid_points = np.column_stack((strike_grid.ravel(), ttm_grid.ravel()))
        vol_surface = rbf(grid_points).reshape(strike_grid.shape)
        
        # S'assurer que les volatilités restent positives
        vol_surface = np.maximum(vol_surface, 5.0)
        
        return vol_surface
    
    def _interpolate_svi(self, strikes: np.ndarray, ttms: np.ndarray, 
                        vols: np.ndarray, strike_grid: np.ndarray, 
                        ttm_grid: np.ndarray) -> np.ndarray:
        """
        Interpolation SVI (Stochastic Volatility Inspired).
        
        CONCEPT: Paramétrise le smile de volatilité avec 5 paramètres
        garantissant l'absence d'arbitrage.
        
        Pour simplifier, on utilise RBF ici mais avec des contraintes.
        """
        # Version simplifiée - utiliser RBF avec post-processing
        vol_surface = self._interpolate_rbf(
            strikes, ttms, vols, strike_grid, ttm_grid
        )
        
        # Appliquer des contraintes pour éviter l'arbitrage
        vol_surface = self._enforce_no_arbitrage_constraints(
            strike_grid, ttm_grid, vol_surface
        )
        
        return vol_surface
    
    def _enforce_no_arbitrage_constraints(self, strike_grid: np.ndarray, 
                                         ttm_grid: np.ndarray, 
                                         vol_surface: np.ndarray) -> np.ndarray:
        """
        Applique des contraintes pour éviter l'arbitrage.
        
        CONDITIONS SANS ARBITRAGE:
        1. σ(K,T) > 0 (évident)
        2. ∂σ/∂T ≥ 0 (la vol augmente généralement avec le temps)
        3. Convexité en K (butterfly spread positif)
        """
        # Contrainte 1: Volatilité positive
        vol_surface = np.maximum(vol_surface, 5.0)
        
        # Contrainte 2: Croissance en temps (approximative)
        for i in range(vol_surface.shape[0] - 1):
            for j in range(vol_surface.shape[1]):
                if vol_surface[i+1, j] < vol_surface[i, j]:
                    # Lisser la transition
                    vol_surface[i+1, j] = vol_surface[i, j] * 1.01
        
        # Contrainte 3: Limiter les pentes extrêmes
        for i in range(vol_surface.shape[0]):
            # Différences finies pour détecter les pentes extrêmes
            if vol_surface.shape[1] > 2:
                diffs = np.diff(vol_surface[i, :])
                max_slope = 0.5  # Max 50% de changement entre points adjacents
                
                for j in range(len(diffs)):
                    if abs(diffs[j]) > max_slope * vol_surface[i, j]:
                        # Adoucir la pente
                        vol_surface[i, j+1] = vol_surface[i, j] * (1 + np.sign(diffs[j]) * max_slope)
        
        return vol_surface
    
    def _calculate_surface_metrics(self, daily_data: pd.DataFrame, 
                                  strikes: np.ndarray, ttms: np.ndarray, 
                                  vols: np.ndarray, spot: float) -> Dict:
        """
        Calcule des métriques importantes de la surface.
        
        MÉTRIQUES CLÉS:
        1. ATM volatility: Vol à la monnaie (référence principale)
        2. Skew: Asymétrie du smile (mesure de peur du marché)
        3. Term structure: Pente temporelle (anticipations)
        4. Butterfly: Convexité du smile (coût des tail risks)
        """
        moneyness = strikes / spot
        
        # 1. ATM Volatility (moneyness ≈ 1)
        atm_mask = (moneyness >= 0.95) & (moneyness <= 1.05)
        atm_vol = vols[atm_mask].mean() if atm_mask.any() else vols.mean()
        
        # 2. 25-Delta Skew (différence entre 25Δ put et call)
        # Approximation: 90% et 110% moneyness
        put_mask = (moneyness >= 0.85) & (moneyness <= 0.95)
        call_mask = (moneyness >= 1.05) & (moneyness <= 1.15)
        
        put_vol = vols[put_mask].mean() if put_mask.any() else np.nan
        call_vol = vols[call_mask].mean() if call_mask.any() else np.nan
        skew_25d = put_vol - call_vol if not (np.isnan(put_vol) or np.isnan(call_vol)) else 0
        
        # 3. Term Structure (pente 3m vs 1y)
        short_mask = (ttms >= 0.2) & (ttms <= 0.3)  # ~3 mois
        long_mask = (ttms >= 0.9) & (ttms <= 1.1)   # ~1 an
        
        short_vol = vols[short_mask].mean() if short_mask.any() else np.nan
        long_vol = vols[long_mask].mean() if long_mask.any() else np.nan
        term_slope = long_vol - short_vol if not (np.isnan(short_vol) or np.isnan(long_vol)) else 0
        
        # 4. Butterfly (convexité à 25Δ)
        # Butterfly = σ(K_ATM) - 0.5*(σ(K_25ΔP) + σ(K_25ΔC))
        butterfly = atm_vol - 0.5 * (put_vol + call_vol) if not (np.isnan(put_vol) or np.isnan(call_vol)) else 0
        
        # 5. Vol of Vol (stabilité de la surface)
        vol_of_vol = vols.std()
        
        # 6. Smile Curvature (2ème dérivée au niveau ATM)
        atm_region = (moneyness >= 0.9) & (moneyness <= 1.1)
        if atm_region.sum() >= 3:
            # Fit polynomial pour estimer la courbure
            poly_coef = np.polyfit(moneyness[atm_region], vols[atm_region], 2)
            smile_curvature = 2 * poly_coef[0]  # Coefficient du terme quadratique
        else:
            smile_curvature = 0
        
        return {
            'spot_price': spot,
            'atm_vol': round(atm_vol, 2),
            'skew_25d': round(skew_25d, 2),
            'term_slope': round(term_slope, 2),
            'butterfly_25d': round(butterfly, 2),
            'vol_of_vol': round(vol_of_vol, 2),
            'smile_curvature': round(smile_curvature, 4),
            'min_vol': round(vols.min(), 2),
            'max_vol': round(vols.max(), 2),
            'vol_range': round(vols.max() - vols.min(), 2),
            'n_options': len(daily_data),
            'avg_moneyness_range': round(moneyness.max() - moneyness.min(), 2)
        }
    
    def _check_arbitrage_conditions(self, strike_grid: np.ndarray, 
                                   ttm_grid: np.ndarray, 
                                   vol_surface: np.ndarray, 
                                   spot: float) -> Dict[str, bool]:
        """
        Vérifie les conditions d'absence d'arbitrage.
        
        TYPES D'ARBITRAGE À ÉVITER:
        1. Calendar spread: Prix décroît avec le temps
        2. Butterfly spread: Non-convexité locale
        3. Volatilités négatives
        """
        checks = {
            'positive_vols': True,
            'calendar_arbitrage_free': True,
            'butterfly_arbitrage_free': True,
            'reasonable_values': True
        }
        
        # 1. Vérifier que toutes les vols sont positives
        if (vol_surface <= 0).any():
            checks['positive_vols'] = False
        
        # 2. Vérifier l'arbitrage calendaire (simpliste)
        # Les options plus longues devraient généralement avoir une vol >= courtes
        for j in range(vol_surface.shape[1]):  # Pour chaque strike
            for i in range(vol_surface.shape[0] - 1):
                if vol_surface[i+1, j] < vol_surface[i, j] * 0.95:  # Tolérance 5%
                    checks['calendar_arbitrage_free'] = False
                    break
        
        # 3. Vérifier la convexité locale (butterfly)
        # Pour chaque maturité, vérifier la convexité en strike
        for i in range(vol_surface.shape[0]):
            if vol_surface.shape[1] >= 3:
                # Approximation de la dérivée seconde
                d2_vol = np.diff(vol_surface[i, :], n=2)
                if (d2_vol < -0.1).any():  # Tolérance pour la concavité
                    checks['butterfly_arbitrage_free'] = False
                    break
        
        # 4. Vérifier que les valeurs sont raisonnables
        if (vol_surface > 500).any() or (vol_surface < 1).any():
            checks['reasonable_values'] = False
        
        return checks
    
    def build_all_surfaces(self, options_df: pd.DataFrame, 
                          method: str = 'rbf',
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> Dict:
        """
        Construit toutes les surfaces pour la période donnée.
        
        Args:
            options_df: DataFrame avec toutes les données
            method: Méthode d'interpolation
            start_date: Date de début (optionnel)
            end_date: Date de fin (optionnel)
            
        Returns:
            Dict avec toutes les surfaces construites
        """
        # Obtenir toutes les dates uniques
        all_dates = sorted(options_df['dataDate'].unique())
        
        # Filtrer par période si spécifié
        if start_date:
            all_dates = [d for d in all_dates if d >= start_date]
        if end_date:
            all_dates = [d for d in all_dates if d <= end_date]
        
        logging.info(f"🏗️  Construction de {len(all_dates)} surfaces...")
        
        successful = 0
        failed = []
        
        for date in all_dates:
            try:
                surface = self.build_surface_from_data(
                    options_df, date, method
                )
                if surface:
                    successful += 1
                else:
                    failed.append(date)
            except Exception as e:
                logging.error(f"❌ Erreur pour {date}: {e}")
                failed.append(date)
        
        logging.info(f"✅ Surfaces construites: {successful}/{len(all_dates)}")
        if failed:
            logging.warning(f"⚠️  Échecs: {failed[:5]}...")
        
        return self.surfaces
    
    def get_implied_volatility(self, surface_date: str, strike: float, 
                              time_to_maturity: float) -> float:
        """
        Obtient la volatilité implicite pour un (K,T) donné.
        
        UTILISATION:
        - Pour pricer une option spécifique
        - Pour calculer les Greeks avec la bonne volatilité
        
        Args:
            surface_date: Date de la surface
            strike: Strike de l'option
            time_to_maturity: Temps jusqu'à maturité (années)
            
        Returns:
            Volatilité implicite interpolée
        """
        if surface_date not in self.surfaces:
            raise ValueError(f"Surface non disponible pour {surface_date}")
        
        surface = self.surfaces[surface_date]
        
        # Interpolation bilinéaire sur la surface
        from scipy.interpolate import interp2d
        
        # Créer l'interpolateur
        f = interp2d(
            surface['strike_grid'][0, :],  # Strikes uniques
            surface['ttm_grid'][:, 0],      # TTMs uniques
            surface['vol_surface'],
            kind='linear',
            bounds_error=False,
            fill_value=None
        )
        
        # Obtenir la volatilité
        vol = f(strike, time_to_maturity)[0]
        
        # Vérifier la validité
        if np.isnan(vol) or vol <= 0:
            # Utiliser la vol ATM comme fallback
            vol = surface['metrics']['atm_vol']
        
        return float(vol)
    
    def analyze_surface_dynamics(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Analyse l'évolution temporelle des métriques de surface.
        
        INSIGHTS POSSIBLES:
        - Augmentation du skew = marché plus nerveux
        - Aplatissement de la term structure = incertitude court terme
        - Hausse du butterfly = demande pour la protection tail risk
        """
        metrics_list = []
        
        for date, metrics in sorted(self.surface_metrics.items()):
            if start_date <= date <= end_date:
                metrics_record = {'date': date}
                metrics_record.update(metrics)
                metrics_list.append(metrics_record)
        
        df_metrics = pd.DataFrame(metrics_list)
        df_metrics['date'] = pd.to_datetime(df_metrics['date'])
        df_metrics.set_index('date', inplace=True)
        
        # Calculer des métriques additionnelles
        df_metrics['vol_regime'] = pd.cut(
            df_metrics['atm_vol'],
            bins=[0, 15, 25, 40, 100],
            labels=['Low', 'Normal', 'High', 'Extreme']
        )
        
        # Changements journaliers
        df_metrics['atm_vol_change'] = df_metrics['atm_vol'].diff()
        df_metrics['skew_change'] = df_metrics['skew_25d'].diff()
        
        # Moyennes mobiles
        df_metrics['atm_vol_ma5'] = df_metrics['atm_vol'].rolling(5).mean()
        df_metrics['skew_ma5'] = df_metrics['skew_25d'].rolling(5).mean()
        
        return df_metrics
    
    def export_surface_data(self, surface_date: str, 
                           output_format: str = 'json') -> str:
        """
        Exporte une surface pour utilisation externe.
        
        Args:
            surface_date: Date de la surface
            output_format: Format d'export ('json', 'csv', 'npz')
            
        Returns:
            Chemin du fichier exporté
        """
        if surface_date not in self.surfaces:
            raise ValueError(f"Surface non disponible pour {surface_date}")
        
        surface = self.surfaces[surface_date]
        output_file = f"surface_{surface_date.replace('-', '')}"
        
        if output_format == 'json':
            # Convertir les arrays numpy en listes
            export_data = {
                'date': surface['date'],
                'spot_price': surface['spot_price'],
                'metrics': surface['metrics'],
                'method': surface['method'],
                'arbitrage_free': surface['arbitrage_free'],
                'strike_grid': surface['strike_grid'].tolist(),
                'ttm_grid': surface['ttm_grid'].tolist(),
                'vol_surface': surface['vol_surface'].tolist()
            }
            
            output_file += '.json'
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
                
        elif output_format == 'csv':
            # Format tabulaire
            data_list = []
            for i in range(surface['vol_surface'].shape[0]):
                for j in range(surface['vol_surface'].shape[1]):
                    data_list.append({
                        'date': surface['date'],
                        'strike': surface['strike_grid'][i, j],
                        'ttm': surface['ttm_grid'][i, j],
                        'implied_vol': surface['vol_surface'][i, j],
                        'spot': surface['spot_price']
                    })
            
            df_export = pd.DataFrame(data_list)
            output_file += '.csv'
            df_export.to_csv(output_file, index=False)
            
        elif output_format == 'npz':
            # Format numpy compressé
            output_file += '.npz'
            np.savez_compressed(
                output_file,
                strike_grid=surface['strike_grid'],
                ttm_grid=surface['ttm_grid'],
                vol_surface=surface['vol_surface'],
                spot_price=surface['spot_price'],
                metrics=surface['metrics']
            )
        
        logging.info(f"💾 Surface exportée: {output_file}")
        return output_file


def main():
    """
    Démontre l'utilisation du constructeur de surfaces de volatilité.
    """
    print("="*70)
    print("🏗️  CONSTRUCTEUR DE SURFACES DE VOLATILITÉ")
    print("="*70)
    
    # Charger les données (assumant qu'elles ont été collectées)
    try:
        df = pd.read_csv('data/AAPL/AAPL_options_historical_7d.csv')
        print(f"✅ Données chargées: {len(df)} options")
    except:
        print("❌ Fichier de données non trouvé. Exécutez d'abord le collecteur.")
        return
    
    # Initialiser le constructeur
    builder = VolatilitySurfaceBuilder()
    
    # Construire une surface pour une date
    dates = df['dataDate'].unique()
    if len(dates) > 0:
        test_date = dates[0]
        print(f"\n📊 Construction de la surface pour {test_date}")
        
        # Tester différentes méthodes
        methods = ['linear', 'cubic', 'rbf']
        
        for method in methods:
            print(f"\n🔧 Méthode: {method}")
            surface = builder.build_surface_from_data(df, test_date, method)
            
            if surface:
                print(f"✅ Surface construite avec {surface['n_points']} points")
                print(f"📈 Métriques:")
                for key, value in surface['metrics'].items():
                    print(f"   - {key}: {value}")
                
                print(f"🔍 Vérifications arbitrage:")
                for check, passed in surface['arbitrage_free'].items():
                    status = "✅" if passed else "❌"
                    print(f"   {status} {check}")
    
    # Construire toutes les surfaces
    print("\n🏗️  Construction de toutes les surfaces...")
    all_surfaces = builder.build_all_surfaces(df, method='rbf')
    
    # Analyser la dynamique
    if len(all_surfaces) >= 2:
        print("\n📊 Analyse de la dynamique des surfaces")
        metrics_df = builder.analyze_surface_dynamics(
            min(dates), max(dates)
        )
        
        print("\n📈 Statistiques des métriques:")
        print(metrics_df[['atm_vol', 'skew_25d', 'term_slope', 'butterfly_25d']].describe())
        
        # Exporter une surface
        print("\n💾 Export de surface...")
        export_file = builder.export_surface_data(test_date, 'json')
        print(f"✅ Surface exportée: {export_file}")
    
    print("\n✅ Construction terminée!")
    print("Les surfaces sont prêtes pour le pricing et la visualisation.")


if __name__ == "__main__":
    main()
