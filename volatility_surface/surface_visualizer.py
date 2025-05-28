"""
Prédicteur de volatilité utilisant le Machine Learning
Auteur: Assistant Claude
Date: Mai 2025

Ce module utilise plusieurs approches ML pour prédire la volatilité implicite
et améliorer les surfaces de volatilité pour le pricing d'options et d'autocall.

POURQUOI LE ML POUR LA VOLATILITÉ?
1. Capturer des patterns non-linéaires complexes
2. Incorporer plus de features que les modèles classiques
3. S'adapter aux régimes de marché changeants
4. Améliorer l'interpolation/extrapolation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Any
import shap  # Pour l'interprétabilité

warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class VolatilityPredictor:
    """
    Prédicteur de volatilité utilisant plusieurs modèles ML.
    
    APPROCHE PÉDAGOGIQUE:
    1. Feature engineering: Créer des variables pertinentes
    2. Modèles multiples: Comparer différentes approches
    3. Ensemble learning: Combiner les prédictions
    4. Interprétabilité: Comprendre les décisions du modèle
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        self.feature_names = []
        self.best_model = None
        
        # Paramètres par défaut des modèles
        self.model_params = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42
            },
            'lightgbm': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'random_state': 42
            },
            'neural_network': {
                'hidden_layer_sizes': (128, 64, 32),
                'activation': 'relu',
                'learning_rate': 'adaptive',
                'max_iter': 1000,
                'random_state': 42
            }
        }
    
    def create_features(self, options_df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée des features avancées pour la prédiction de volatilité.
        
        FEATURES CRÉÉES:
        1. Moneyness et transformations
        2. Features temporelles
        3. Métriques de marché
        4. Features d'interaction
        5. Features techniques
        
        Cette étape est CRUCIALE - de bonnes features = bonnes prédictions!
        """
        logging.info("🔧 Création des features...")
        
        df = options_df.copy()
        
        # 1. FEATURES DE BASE
        # Moneyness (K/S) - mesure si l'option est ITM/ATM/OTM
        df['moneyness'] = df['strike'] / df['spotPrice']
        df['log_moneyness'] = np.log(df['moneyness'])
        df['moneyness_squared'] = df['moneyness'] ** 2
        
        # 2. FEATURES TEMPORELLES
        # Le temps influence fortement la volatilité
        df['sqrt_ttm'] = np.sqrt(df['yearsToExpiration'])  # Racine du temps (Black-Scholes)
        df['inv_ttm'] = 1 / (df['yearsToExpiration'] + 0.01)  # Inverse du temps
        df['ttm_squared'] = df['yearsToExpiration'] ** 2
        
        # Catégories de maturité
        df['is_short_term'] = (df['yearsToExpiration'] < 0.25).astype(int)
        df['is_medium_term'] = ((df['yearsToExpiration'] >= 0.25) & 
                                (df['yearsToExpiration'] < 1.0)).astype(int)
        df['is_long_term'] = (df['yearsToExpiration'] >= 1.0).astype(int)
        
        # 3. FEATURES DE LIQUIDITÉ
        # La liquidité affecte la fiabilité des prix
        df['log_volume'] = np.log1p(df['volume'])
        df['log_open_interest'] = np.log1p(df['openInterest'])
        df['liquidity_score'] = df['log_volume'] + df['log_open_interest']
        
        # 4. FEATURES D'ASYMÉTRIE
        # Différence entre calls et puts révèle le sentiment du marché
        df['is_call'] = (df['optionType'] == 'CALL').astype(int)
        df['is_put'] = (df['optionType'] == 'PUT').astype(int)
        
        # 5. FEATURES DE MARCHÉ (si disponibles)
        if 'delta' in df.columns:
            df['abs_delta'] = np.abs(df['delta'])
            df['delta_squared'] = df['delta'] ** 2
        
        if 'gamma' in df.columns:
            df['log_gamma'] = np.log1p(np.abs(df['gamma']))
        
        if 'vega' in df.columns:
            df['log_vega'] = np.log1p(np.abs(df['vega']))
        
        # 6. FEATURES D'INTERACTION
        # Capturer les effets non-linéaires
        df['moneyness_ttm'] = df['moneyness'] * df['yearsToExpiration']
        df['moneyness_sqrt_ttm'] = df['moneyness'] * df['sqrt_ttm']
        
        # 7. FEATURES DE DISTANCE
        # Distance par rapport à des points de référence
        df['distance_from_atm'] = np.abs(df['moneyness'] - 1.0)
        df['distance_from_90'] = np.abs(df['moneyness'] - 0.9)
        df['distance_from_110'] = np.abs(df['moneyness'] - 1.1)
        
        # 8. FEATURES POLYNOMIALES
        # Pour capturer la forme du smile
        df['moneyness_3'] = df['moneyness'] ** 3
        df['moneyness_4'] = df['moneyness'] ** 4
        
        # 9. FEATURES CYCLIQUES (si on a des données temporelles)
        if 'dataDate' in df.columns:
            df['dataDate'] = pd.to_datetime(df['dataDate'])
            df['day_of_week'] = df['dataDate'].dt.dayofweek
            df['day_of_month'] = df['dataDate'].dt.day
            df['month'] = df['dataDate'].dt.month
            
            # Encoding cyclique pour capturer la périodicité
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # 10. STATISTIQUES ROULANTES (si on a l'historique)
        # Ces features capturent la dynamique temporelle
        if 'dataDate' in df.columns:
            df = df.sort_values(['ticker', 'dataDate', 'strike', 'expirationDate'])
            
            # Volatilité moyenne sur les 5 derniers jours (par ticker)
            df['vol_ma5'] = df.groupby('ticker')['impliedVolatility'].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean()
            )
            
            # Écart par rapport à la moyenne
            df['vol_deviation'] = df['impliedVolatility'] - df['vol_ma5']
        
        # Sélectionner les features finales
        feature_cols = [
            'moneyness', 'log_moneyness', 'moneyness_squared', 'moneyness_3',
            'yearsToExpiration', 'sqrt_ttm', 'inv_ttm', 'ttm_squared',
            'is_short_term', 'is_medium_term', 'is_long_term',
            'log_volume', 'log_open_interest', 'liquidity_score',
            'is_call', 'is_put',
            'moneyness_ttm', 'moneyness_sqrt_ttm',
            'distance_from_atm', 'distance_from_90', 'distance_from_110'
        ]
        
        # Ajouter les features optionnelles si elles existent
        optional_features = [
            'abs_delta', 'delta_squared', 'log_gamma', 'log_vega',
            'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos',
            'vol_ma5', 'vol_deviation'
        ]
        
        for feat in optional_features:
            if feat in df.columns:
                feature_cols.append(feat)
        
        self.feature_names = feature_cols
        
        # Nettoyer les valeurs manquantes
        df[feature_cols] = df[feature_cols].fillna(0)
        
        # Remplacer les infinis par des valeurs grandes mais finies
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], [999, -999])
        
        logging.info(f"✅ {len(feature_cols)} features créées")
        
        return df
    
    def prepare_data(self, df: pd.DataFrame, 
                    target_col: str = 'impliedVolatility',
                    test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, 
                                                     np.ndarray, np.ndarray]:
        """
        Prépare les données pour l'entraînement.
        
        CONSIDÉRATIONS IMPORTANTES:
        1. Split temporel (pas aléatoire!) pour les séries temporelles
        2. Normalisation robuste aux outliers
        3. Gestion des valeurs manquantes
        """
        # S'assurer que les features sont créées
        if not all(col in df.columns for col in self.feature_names):
            df = self.create_features(df)
        
        # Features et target
        X = df[self.feature_names].values
        y = df[target_col].values
        
        # Split temporel si on a des dates
        if 'dataDate' in df.columns:
            # Trier par date
            df_sorted = df.sort_values('dataDate')
            split_idx = int(len(df_sorted) * (1 - test_size))
            
            X_train = X[:split_idx]
            X_test = X[split_idx:]
            y_train = y[:split_idx]
            y_test = y[split_idx:]
            
            logging.info("📅 Split temporel utilisé")
        else:
            # Split aléatoire si pas de dates
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            logging.info("🎲 Split aléatoire utilisé")
        
        logging.info(f"📊 Train: {len(X_train)}, Test: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, model_name: str, X_train: np.ndarray, 
                   y_train: np.ndarray, X_val: Optional[np.ndarray] = None,
                   y_val: Optional[np.ndarray] = None) -> Any:
        """
        Entraîne un modèle spécifique.
        
        MODÈLES DISPONIBLES:
        1. Random Forest: Robuste, peu de tuning nécessaire
        2. Gradient Boosting: Précis mais plus lent
        3. XGBoost: Version optimisée du GB
        4. LightGBM: Très rapide, bon pour grandes données
        5. Neural Network: Capture les patterns complexes
        """
        logging.info(f"🚀 Entraînement du modèle {model_name}...")
        
        # Créer le scaler
        scaler = RobustScaler()  # Robuste aux outliers
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Sauvegarder le scaler
        self.scalers[model_name] = scaler
        
        # Créer et entraîner le modèle
        if model_name == 'random_forest':
            model = RandomForestRegressor(**self.model_params[model_name])
            model.fit(X_train_scaled, y_train)
            
        elif model_name == 'gradient_boosting':
            model = GradientBoostingRegressor(**self.model_params[model_name])
            model.fit(X_train_scaled, y_train)
            
        elif model_name == 'xgboost':
            model = xgb.XGBRegressor(**self.model_params[model_name])
            eval_set = [(X_train_scaled, y_train)]
            if X_val is not None:
                X_val_scaled = scaler.transform(X_val)
                eval_set.append((X_val_scaled, y_val))
            
            model.fit(
                X_train_scaled, y_train,
                eval_set=eval_set,
                early_stopping_rounds=10,
                verbose=False
            )
            
        elif model_name == 'lightgbm':
            model = lgb.LGBMRegressor(**self.model_params[model_name])
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_train_scaled, y_train)],
                callbacks=[lgb.log_evaluation(0)]
            )
            
        elif model_name == 'neural_network':
            # Normaliser la target aussi pour le NN
            y_scaler = StandardScaler()
            y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
            
            model = MLPRegressor(**self.model_params[model_name])
            model.fit(X_train_scaled, y_train_scaled)
            
            # Créer un wrapper pour dé-normaliser les prédictions
            class NNWrapper:
                def __init__(self, model, y_scaler):
                    self.model = model
                    self.y_scaler = y_scaler
                
                def predict(self, X):
                    y_pred_scaled = self.model.predict(X)
                    return self.y_scaler.inverse_transform(
                        y_pred_scaled.reshape(-1, 1)
                    ).ravel()
                
                def __getattr__(self, name):
                    return getattr(self.model, name)
            
            model = NNWrapper(model, y_scaler)
        
        else:
            raise ValueError(f"Modèle inconnu: {model_name}")
        
        self.models[model_name] = model
        logging.info(f"✅ Modèle {model_name} entraîné")
        
        return model
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: Optional[np.ndarray] = None,
                        y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Entraîne tous les modèles disponibles.
        
        STRATÉGIE:
        1. Entraîner chaque modèle
        2. Comparer les performances
        3. Sélectionner le meilleur
        4. Optionnel: Créer un ensemble
        """
        logging.info("🎯 Entraînement de tous les modèles...")
        
        for model_name in self.model_params.keys():
            try:
                self.train_model(model_name, X_train, y_train, X_val, y_val)
            except Exception as e:
                logging.error(f"❌ Erreur pour {model_name}: {e}")
        
        return self.models
    
    def evaluate_model(self, model_name: str, X_test: np.ndarray, 
                      y_test: np.ndarray) -> Dict[str, float]:
        """
        Évalue un modèle sur les données de test.
        
        MÉTRIQUES:
        1. MSE: Erreur quadratique moyenne
        2. MAE: Erreur absolue moyenne
        3. R²: Coefficient de détermination
        4. MAPE: Erreur en pourcentage
        """
        if model_name not in self.models:
            raise ValueError(f"Modèle {model_name} non entraîné")
        
        model = self.models[model_name]
        scaler = self.scalers[model_name]
        
        # Prédire
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        
        # Calculer les métriques
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        # Éviter la division par zéro
        mask = y_test > 0
        if mask.any():
            mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
        else:
            mape = np.nan
        
        metrics = {
            'mse': round(mse, 4),
            'rmse': round(np.sqrt(mse), 4),
            'mae': round(mae, 4),
            'r2': round(r2, 4),
            'mape': round(mape, 2) if not np.isnan(mape) else None
        }
        
        self.performance_metrics[model_name] = metrics
        
        return metrics
    
    def evaluate_all_models(self, X_test: np.ndarray, 
                           y_test: np.ndarray) -> pd.DataFrame:
        """
        Évalue tous les modèles et compare leurs performances.
        
        ANALYSE:
        - Quel modèle performe le mieux?
        - Y a-t-il du surapprentissage?
        - Les erreurs sont-elles acceptables?
        """
        logging.info("📊 Évaluation de tous les modèles...")
        
        results = []
        
        for model_name in self.models.keys():
            try:
                metrics = self.evaluate_model(model_name, X_test, y_test)
                metrics['model'] = model_name
                results.append(metrics)
                
                logging.info(f"{model_name}: RMSE={metrics['rmse']}, R²={metrics['r2']}")
                
            except Exception as e:
                logging.error(f"❌ Erreur évaluation {model_name}: {e}")
        
        # Créer un DataFrame de comparaison
        df_results = pd.DataFrame(results)
        df_results = df_results.set_index('model')
        
        # Identifier le meilleur modèle (basé sur RMSE)
        self.best_model = df_results['rmse'].idxmin()
        logging.info(f"🏆 Meilleur modèle: {self.best_model}")
        
        return df_results
    
    def get_feature_importance(self, model_name: str, 
                              n_features: int = 20) -> pd.DataFrame:
        """
        Obtient l'importance des features pour un modèle.
        
        INTERPRÉTABILITÉ:
        - Quelles features sont les plus importantes?
        - Y a-t-il des features surprenantes?
        - Peut-on simplifier le modèle?
        """
        if model_name not in self.models:
            raise ValueError(f"Modèle {model_name} non disponible")
        
        model = self.models[model_name]
        
        # Obtenir l'importance selon le type de modèle
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
        elif model_name == 'neural_network':
            # Pour NN, on peut utiliser la permutation importance
            logging.info("NN: Utilisation de l'importance par permutation")
            return pd.DataFrame()  # Simplification pour cet exemple
        else:
            return pd.DataFrame()
        
        # Créer un DataFrame
        df_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        })
        
        # Trier par importance
        df_importance = df_importance.sort_values('importance', ascending=False)
        
        # Normaliser (somme = 100%)
        df_importance['importance_pct'] = (
            df_importance['importance'] / df_importance['importance'].sum() * 100
        )
        
        self.feature_importance[model_name] = df_importance
        
        return df_importance.head(n_features)
    
    def predict_volatility(self, X: np.ndarray, 
                          model_name: Optional[str] = None,
                          return_uncertainty: bool = False) -> np.ndarray:
        """
        Prédit la volatilité pour de nouvelles données.
        
        Args:
            X: Features (même format que l'entraînement)
            model_name: Modèle à utiliser (défaut: meilleur)
            return_uncertainty: Retourner l'incertitude (si disponible)
            
        Returns:
            Prédictions de volatilité
        """
        if model_name is None:
            model_name = self.best_model or list(self.models.keys())[0]
        
        if model_name not in self.models:
            raise ValueError(f"Modèle {model_name} non disponible")
        
        model = self.models[model_name]
        scaler = self.scalers[model_name]
        
        # Scaler les features
        X_scaled = scaler.transform(X)
        
        # Prédire
        predictions = model.predict(X_scaled)
        
        # Optionnel: Obtenir l'incertitude (pour RF)
        if return_uncertainty and model_name == 'random_forest':
            # Prédictions de chaque arbre
            tree_predictions = np.array([
                tree.predict(X_scaled) for tree in model.estimators_
            ])
            
            # Écart-type comme mesure d'incertitude
            uncertainty = np.std(tree_predictions, axis=0)
            
            return predictions, uncertainty
        
        return predictions
    
    def create_ensemble_predictor(self, weights: Optional[Dict[str, float]] = None) -> 'EnsemblePredictor':
        """
        Crée un prédicteur ensemble combinant plusieurs modèles.
        
        AVANTAGES DE L'ENSEMBLE:
        1. Réduit le surapprentissage
        2. Plus robuste
        3. Capture différents aspects des données
        """
        if not self.models:
            raise ValueError("Aucun modèle entraîné")
        
        if weights is None:
            # Poids basés sur les performances (inversement proportionnel à RMSE)
            if self.performance_metrics:
                rmse_values = {
                    model: metrics['rmse'] 
                    for model, metrics in self.performance_metrics.items()
                }
                
                # Inverser et normaliser
                inv_rmse = {k: 1/v for k, v in rmse_values.items()}
                total = sum(inv_rmse.values())
                weights = {k: v/total for k, v in inv_rmse.items()}
            else:
                # Poids égaux par défaut
                n_models = len(self.models)
                weights = {model: 1/n_models for model in self.models.keys()}
        
        return EnsemblePredictor(self, weights)
    
    def explain_predictions(self, X: np.ndarray, model_name: Optional[str] = None,
                           n_samples: int = 100) -> Dict:
        """
        Explique les prédictions en utilisant SHAP.
        
        POURQUOI L'EXPLICABILITÉ?
        - Comprendre les décisions du modèle
        - Identifier les biais potentiels
        - Gagner la confiance des utilisateurs
        """
        if model_name is None:
            model_name = self.best_model
        
        if model_name not in self.models:
            raise ValueError(f"Modèle {model_name} non disponible")
        
        model = self.models[model_name]
        scaler = self.scalers[model_name]
        
        # Préparer les données
        X_scaled = scaler.transform(X[:n_samples])
        
        # Créer l'explainer SHAP
        if model_name in ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
        else:
            # Pour les autres modèles, utiliser KernelExplainer (plus lent)
            explainer = shap.KernelExplainer(model.predict, X_scaled[:100])
            shap_values = explainer.shap_values(X_scaled)
        
        # Calculer l'importance moyenne
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'shap_importance': mean_abs_shap
        }).sort_values('shap_importance', ascending=False)
        
        return {
            'shap_values': shap_values,
            'feature_importance': importance_df,
            'explainer': explainer
        }
    
    def save_models(self, path_prefix: str = 'models/volatility_predictor'):
        """Sauvegarde tous les modèles entraînés."""
        import os
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        
        # Sauvegarder les modèles
        for model_name, model in self.models.items():
            model_path = f"{path_prefix}_{model_name}.pkl"
            joblib.dump(model, model_path)
            
            # Sauvegarder le scaler associé
            scaler_path = f"{path_prefix}_{model_name}_scaler.pkl"
            joblib.dump(self.scalers[model_name], scaler_path)
        
        # Sauvegarder les métadonnées
        metadata = {
            'feature_names': self.feature_names,
            'best_model': self.best_model,
            'performance_metrics': self.performance_metrics
        }
        
        metadata_path = f"{path_prefix}_metadata.pkl"
        joblib.dump(metadata, metadata_path)
        
        logging.info(f"✅ Modèles sauvegardés avec préfixe: {path_prefix}")
    
    def load_models(self, path_prefix: str = 'models/volatility_predictor'):
        """Charge les modèles sauvegardés."""
        # Charger les métadonnées
        metadata_path = f"{path_prefix}_metadata.pkl"
        metadata = joblib.load(metadata_path)
        
        self.feature_names = metadata['feature_names']
        self.best_model = metadata['best_model']
        self.performance_metrics = metadata['performance_metrics']
        
        # Charger les modèles
        for model_name in metadata['performance_metrics'].keys():
            model_path = f"{path_prefix}_{model_name}.pkl"
            scaler_path = f"{path_prefix}_{model_name}_scaler.pkl"
            
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
                self.scalers[model_name] = joblib.load(scaler_path)
        
        logging.info(f"✅ {len(self.models)} modèles chargés")


class EnsemblePredictor:
    """
    Prédicteur ensemble combinant plusieurs modèles.
    
    STRATÉGIES D'ENSEMBLE:
    1. Moyenne pondérée simple
    2. Stacking (meta-learning)
    3. Blending
    """
    
    def __init__(self, predictor: VolatilityPredictor, weights: Dict[str, float]):
        self.predictor = predictor
        self.weights = weights
        self.models = {k: v for k, v in predictor.models.items() if k in weights}
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédit en combinant tous les modèles."""
        predictions = []
        weights_list = []
        
        for model_name, weight in self.weights.items():
            if model_name in self.models:
                pred = self.predictor.predict_volatility(X, model_name)
                predictions.append(pred)
                weights_list.append(weight)
        
        # Moyenne pondérée
        predictions = np.array(predictions)
        weights_array = np.array(weights_list).reshape(-1, 1)
        
        ensemble_pred = np.sum(predictions * weights_array, axis=0)
        
        return ensemble_pred
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prédit avec une estimation de l'incertitude."""
        predictions = []
        
        for model_name in self.models.keys():
            pred = self.predictor.predict_volatility(X, model_name)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Moyenne et écart-type
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred


def demonstrate_volatility_prediction():
    """
    Démontre l'utilisation complète du prédicteur de volatilité.
    """
    print("="*70)
    print("🤖 PRÉDICTEUR ML DE VOLATILITÉ")
    print("="*70)
    
    # Charger les données
    try:
        df = pd.read_csv('data/AAPL/AAPL_options_historical_7d.csv')
        print(f"✅ Données chargées: {len(df)} options")
    except:
        print("❌ Fichier de données non trouvé.")
        print("Génération de données synthétiques pour la démonstration...")
        
        # Générer des données synthétiques pour la démo
        np.random.seed(42)
        n_samples = 1000
        
        df = pd.DataFrame({
            'ticker': 'AAPL',
            'spotPrice': 150,
            'strike': np.random.uniform(120, 180, n_samples),
            'yearsToExpiration': np.random.uniform(0.05, 2, n_samples),
            'volume': np.random.lognormal(5, 2, n_samples),
            'openInterest': np.random.lognormal(6, 2, n_samples),
            'optionType': np.random.choice(['CALL', 'PUT'], n_samples),
            'delta': np.random.uniform(-1, 1, n_samples),
            'gamma': np.random.uniform(0, 0.1, n_samples),
            'vega': np.random.uniform(0, 1, n_samples)
        })
        
        # Générer une volatilité réaliste
        moneyness = df['strike'] / df['spotPrice']
        base_vol = 20 + 5 * np.abs(np.log(moneyness))  # Smile effect
        noise = np.random.normal(0, 2, n_samples)
        df['impliedVolatility'] = np.maximum(5, base_vol + noise)
    
    # Initialiser le prédicteur
    predictor = VolatilityPredictor()
    
    # Créer les features
    print("\n🔧 Création des features...")
    df_features = predictor.create_features(df)
    print(f"✅ {len(predictor.feature_names)} features créées")
    
    # Préparer les données
    print("\n📊 Préparation des données...")
    X_train, X_test, y_train, y_test = predictor.prepare_data(df_features)
    
    # Entraîner tous les modèles
    print("\n🚀 Entraînement des modèles...")
    predictor.train_all_models(X_train, y_train)
    
    # Évaluer les modèles
    print("\n📈 Évaluation des modèles...")
    results_df = predictor.evaluate_all_models(X_test, y_test)
    print("\nPerformances des modèles:")
    print(results_df)
    
    # Feature importance du meilleur modèle
    if predictor.best_model in ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']:
        print(f"\n🔍 Features importantes ({predictor.best_model}):")
        importance_df = predictor.get_feature_importance(predictor.best_model, n_features=10)
        print(importance_df[['feature', 'importance_pct']].to_string(index=False))
    
    # Créer un ensemble
    print("\n🎭 Création d'un prédicteur ensemble...")
    ensemble = predictor.create_ensemble_predictor()
    
    # Exemple de prédiction
    print("\n🔮 Exemple de prédiction:")
    # Prendre quelques exemples du test set
    X_sample = X_test[:5]
    y_true_sample = y_test[:5]
    
    # Prédictions individuelles
    for model_name in list(predictor.models.keys())[:3]:  # Top 3 modèles
        preds = predictor.predict_volatility(X_sample, model_name)
        print(f"\n{model_name}:")
        for i in range(len(preds)):
            print(f"  Vraie vol: {y_true_sample[i]:.1f}%, Prédite: {preds[i]:.1f}%")
    
    # Prédiction ensemble
    ensemble_preds = ensemble.predict(X_sample)
    print("\nEnsemble:")
    for i in range(len(ensemble_preds)):
        print(f"  Vraie vol: {y_true_sample[i]:.1f}%, Prédite: {ensemble_preds[i]:.1f}%")
    
    # Sauvegarder les modèles
    print("\n💾 Sauvegarde des modèles...")
    predictor.save_models()
    
    print("\n✅ Démonstration terminée!")
    print("Les modèles sont prêts pour prédire la volatilité dans le pricing d'options.")


if __name__ == "__main__":
    demonstrate_volatility_prediction()
