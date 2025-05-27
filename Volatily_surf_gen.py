"""
Transformateur de données d'options et générateur de surfaces de volatilité
Auteur: Assistant Claude
Date: Mai 2025

Ce module transforme les données brutes d'options en format structuré
et génère des surfaces de volatilité implicite 3D pour chaque jour.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.interpolate import griddata, RBFInterpolator
from scipy.spatial import ConvexHull
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class OptionsDataTransformer:
    """
    Classe pour transformer les données d'options en format structuré
    et créer des surfaces de volatilité
    """
    
    def __init__(self):
        self.structured_data = None
        self.volatility_surfaces = {}
        
    def transform_to_structured_format(self, raw_options_df, price_col='lastPrice', 
                                     vol_col='impliedVolatilityPct', date_col='dataDate'):
        """
        Transforme les données brutes d'options en format structuré avec index hiérarchique
        
        Args:
            raw_options_df: DataFrame avec les données brutes d'options
            price_col: Nom de la colonne prix
            vol_col: Nom de la colonne volatilité implicite
            date_col: Nom de la colonne date
            
        Returns:
            DataFrame avec index hiérarchique (Date, Option_ID)
        """
        print("🔄 Transformation des données en format structuré...")
        
        # Copier et nettoyer les données
        df = raw_options_df.copy()
        
        # Créer un identifiant unique pour chaque option
        df['Option_ID'] = (
            df['optionType'] + '_' + 
            df['strike'].astype(str) + '_' + 
            df['expirationDate'].astype(str)
        )
        
        # Sélectionner les colonnes essentielles
        essential_cols = [
            date_col, 'Option_ID', 'optionType', 'strike', 'expirationDate',
            price_col, 'yearsToExpiration', 'theta', vol_col,
            'underlyingPrice', 'delta', 'gamma', 'vega'
        ]
        
        # Vérifier que toutes les colonnes existent
        available_cols = [col for col in essential_cols if col in df.columns]
        missing_cols = [col for col in essential_cols if col not in df.columns]
        
        if missing_cols:
            print(f"⚠️  Colonnes manquantes (seront ignorées): {missing_cols}")
        
        df_clean = df[available_cols].copy()
        
        # Renommer les colonnes pour la clarté
        column_mapping = {
            price_col: 'Prix_Option',
            'yearsToExpiration': 'TTM',
            'strike': 'K',
            'theta': 'Theta',
            vol_col: 'Vol_Implicite',
            'underlyingPrice': 'Prix_Sous_Jacent',
            'optionType': 'Type_Option',
            'expirationDate': 'Date_Expiration'
        }
        
        df_clean.rename(columns=column_mapping, inplace=True)
        
        # Convertir les dates
        if date_col in df_clean.columns:
            df_clean[date_col] = pd.to_datetime(df_clean[date_col])
        
        # Créer l'index hiérarchique
        df_structured = df_clean.set_index([date_col, 'Option_ID'])
        
        # Trier par date puis par option
        df_structured.sort_index(inplace=True)
        
        print(f"✅ Transformation terminée:")
        print(f"   - Dates uniques: {df_structured.index.get_level_values(0).nunique()}")
        print(f"   - Options par jour (moyenne): {len(df_structured) / df_structured.index.get_level_values(0).nunique():.0f}")
        print(f"   - Période couverte: {df_structured.index.get_level_values(0).min()} à {df_structured.index.get_level_values(0).max()}")
        
        self.structured_data = df_structured
        return df_structured
    
    def create_daily_volatility_surface(self, date, min_points=10, interpolation_method='cubic'):
        """
        Crée une surface de volatilité pour une date donnée
        
        Args:
            date: Date pour laquelle créer la surface
            min_points: Nombre minimum de points requis
            interpolation_method: Méthode d'interpolation ('linear', 'cubic', 'rbf')
            
        Returns:
            Dict avec les données de la surface
        """
        if self.structured_data is None:
            raise ValueError("Données structurées non disponibles. Exécutez d'abord transform_to_structured_format()")
        
        # Extraire les données pour cette date
        try:
            daily_data = self.structured_data.loc[date].copy()
        except KeyError:
            print(f"❌ Aucune donnée pour la date {date}")
            return None
        
        if len(daily_data) < min_points:
            print(f"⚠️  Pas assez de points pour {date} ({len(daily_data)} < {min_points})")
            return None
        
        print(f"📊 Création de la surface pour {date} ({len(daily_data)} options)")
        
        # Préparer les données pour l'interpolation
        strikes = daily_data['K'].values
        ttms = daily_data['TTM'].values
        vols = daily_data['Vol_Implicite'].values
        
        # Filtrer les valeurs aberrantes
        vol_mask = (vols > 1) & (vols < 200)  # Entre 1% et 200%
        ttm_mask = ttms > 0.01  # Au moins quelques jours
        
        mask = vol_mask & ttm_mask
        if mask.sum() < min_points:
            print(f"⚠️  Pas assez de points valides après filtrage ({mask.sum()} < {min_points})")
            return None
        
        strikes_clean = strikes[mask]
        ttms_clean = ttms[mask]
        vols_clean = vols[mask]
        
        # Créer la grille pour l'interpolation
        strike_min, strike_max = strikes_clean.min(), strikes_clean.max()
        ttm_min, ttm_max = ttms_clean.min(), ttms_clean.max()
        
        # Grille plus dense pour un rendu lisse
        strike_grid = np.linspace(strike_min, strike_max, 30)
        ttm_grid = np.linspace(ttm_min, ttm_max, 20)
        strike_mesh, ttm_mesh = np.meshgrid(strike_grid, ttm_grid)
        
        # Interpolation de la volatilité
        points = np.column_stack((strikes_clean, ttms_clean))
        
        try:
            if interpolation_method == 'rbf':
                # Interpolation RBF (plus lisse pour les surfaces de volatilité)
                rbf = RBFInterpolator(points, vols_clean, kernel='thin_plate_spline')
                vol_mesh = rbf(np.column_stack((strike_mesh.ravel(), ttm_mesh.ravel())))
                vol_mesh = vol_mesh.reshape(strike_mesh.shape)
            else:
                # Interpolation griddata
                vol_mesh = griddata(
                    points, vols_clean, 
                    (strike_mesh, ttm_mesh), 
                    method=interpolation_method, 
                    fill_value=np.nan
                )
        except Exception as e:
            print(f"❌ Erreur d'interpolation: {e}")
            return None
        
        # Calculer des métriques de surface
        surface_metrics = self._calculate_surface_metrics(
            daily_data, strikes_clean, ttms_clean, vols_clean
        )
        
        surface_data = {
            'date': date,
            'strike_mesh': strike_mesh,
            'ttm_mesh': ttm_mesh,
            'vol_mesh': vol_mesh,
            'raw_data': daily_data,
            'clean_strikes': strikes_clean,
            'clean_ttms': ttms_clean,
            'clean_vols': vols_clean,
            'metrics': surface_metrics,
            'n_points': len(daily_data)
        }
        
        return surface_data
    
    def _calculate_surface_metrics(self, daily_data, strikes, ttms, vols):
        """
        Calcule des métriques utiles pour la surface de volatilité
        """
        # Prix du sous-jacent
        spot_price = daily_data['Prix_Sous_Jacent'].iloc[0]
        
        # Calculer la moneyness
        moneyness = strikes / spot_price
        
        # ATM volatility (volatilité près de la monnaie)
        atm_mask = (moneyness >= 0.95) & (moneyness <= 1.05)
        atm_vol = vols[atm_mask].mean() if atm_mask.any() else vols.mean()
        
        # Skew (différence vol puts OTM vs calls OTM)
        otm_put_mask = moneyness < 0.9
        otm_call_mask = moneyness > 1.1
        
        otm_put_vol = vols[otm_put_mask].mean() if otm_put_mask.any() else np.nan
        otm_call_vol = vols[otm_call_mask].mean() if otm_call_mask.any() else np.nan
        skew = otm_put_vol - otm_call_vol if not (np.isnan(otm_put_vol) or np.isnan(otm_call_vol)) else np.nan
        
        # Term structure (pente de volatilité vs maturité)
        short_term_mask = ttms < 0.25  # < 3 mois
        long_term_mask = ttms > 0.5    # > 6 mois
        
        short_term_vol = vols[short_term_mask].mean() if short_term_mask.any() else np.nan
        long_term_vol = vols[long_term_mask].mean() if long_term_mask.any() else np.nan
        term_structure = long_term_vol - short_term_vol if not (np.isnan(short_term_vol) or np.isnan(long_term_vol)) else np.nan
        
        return {
            'spot_price': spot_price,
            'atm_vol': atm_vol,
            'vol_range': [vols.min(), vols.max()],
            'vol_std': vols.std(),
            'skew': skew,
            'term_structure': term_structure,
            'strike_range': [strikes.min(), strikes.max()],
            'ttm_range': [ttms.min(), ttms.max()]
        }
    
    def create_all_daily_surfaces(self, start_date=None, end_date=None, save_progress=True):
        """
        Crée les surfaces de volatilité pour toutes les dates disponibles
        
        Args:
            start_date: Date de début (optionnel)
            end_date: Date de fin (optionnel)
            save_progress: Sauvegarder les résultats au fur et à mesure
            
        Returns:
            Dict avec toutes les surfaces {date: surface_data}
        """
        if self.structured_data is None:
            raise ValueError("Données structurées non disponibles")
        
        # Obtenir toutes les dates uniques
        all_dates = self.structured_data.index.get_level_values(0).unique()
        
        # Filtrer par période si spécifié
        if start_date:
            all_dates = all_dates[all_dates >= pd.to_datetime(start_date)]
        if end_date:
            all_dates = all_dates[all_dates <= pd.to_datetime(end_date)]
        
        print(f"🔄 Création des surfaces pour {len(all_dates)} dates...")
        
        surfaces = {}
        failed_dates = []
        
        for i, date in enumerate(all_dates):
            print(f"📊 Traitement {i+1}/{len(all_dates)}: {date.strftime('%Y-%m-%d')}")
            
            surface = self.create_daily_volatility_surface(date)
            
            if surface is not None:
                surfaces[date] = surface
                print(f"   ✅ Surface créée ({surface['n_points']} points)")
            else:
                failed_dates.append(date)
                print(f"   ❌ Échec création surface")
        
        print(f"\n🎉 Processus terminé:")
        print(f"   - Surfaces créées: {len(surfaces)}")
        print(f"   - Échecs: {len(failed_dates)}")
        
        if failed_dates:
            print(f"   - Dates échouées: {[d.strftime('%Y-%m-%d') for d in failed_dates[:5]]}")
        
        self.volatility_surfaces = surfaces
        return surfaces
    
    def visualize_surface_3d(self, date, save_fig=True, fig_size=(12, 8)):
        """
        Visualise une surface de volatilité en 3D avec Matplotlib
        
        Args:
            date: Date de la surface à visualiser
            save_fig: Sauvegarder la figure
            fig_size: Taille de la figure
        """
        if date not in self.volatility_surfaces:
            print(f"❌ Surface non trouvée pour {date}")
            return None
        
        surface = self.volatility_surfaces[date]
        
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111, projection='3d')
        
        # Surface principale
        surf = ax.plot_surface(
            surface['strike_mesh'], 
            surface['ttm_mesh'], 
            surface['vol_mesh'],
            cmap='viridis', 
            alpha=0.8,
            linewidth=0.5,
            edgecolors='white'
        )
        
        # Points de données réels
        ax.scatter(
            surface['clean_strikes'], 
            surface['clean_ttms'], 
            surface['clean_vols'],
            c='red', 
            s=20, 
            alpha=0.8,
            label='Données réelles'
        )
        
        # Labels et titre
        ax.set_xlabel('Strike (K)', fontsize=10)
        ax.set_ylabel('Time to Maturity (années)', fontsize=10)
        ax.set_zlabel('Volatilité Implicite (%)', fontsize=10)
        ax.set_title(f'Surface de Volatilité - {date.strftime("%Y-%m-%d")}\n'
                    f'Spot: ${surface["metrics"]["spot_price"]:.2f} | '
                    f'ATM Vol: {surface["metrics"]["atm_vol"]:.1f}%', 
                    fontsize=12)
        
        # Colorbar
        fig.colorbar(surf, shrink=0.5, aspect=5, label='Vol Implicite (%)')
        ax.legend()
        
        plt.tight_layout()
        
        if save_fig:
            filename = f'volatility_surface_{date.strftime("%Y%m%d")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Figure sauvegardée: {filename}")
        
        plt.show()
        return fig
    
    def visualize_surface_interactive(self, date):
        """
        Visualise une surface de volatilité interactive avec Plotly
        
        Args:
            date: Date de la surface à visualiser
        """
        if date not in self.volatility_surfaces:
            print(f"❌ Surface non trouvée pour {date}")
            return None
        
        surface = self.volatility_surfaces[date]
        
        # Surface 3D interactive
        fig = go.Figure(data=[
            go.Surface(
                x=surface['strike_mesh'],
                y=surface['ttm_mesh'],
                z=surface['vol_mesh'],
                colorscale='Viridis',
                opacity=0.8,
                name='Surface Vol'
            )
        ])
        
        # Ajouter les points de données réels
        fig.add_trace(
            go.Scatter3d(
                x=surface['clean_strikes'],
                y=surface['clean_ttms'],
                z=surface['clean_vols'],
                mode='markers',
                marker=dict(
                    size=4,
                    color='red',
                    symbol='circle'
                ),
                name='Données réelles'
            )
        )
        
        # Mise en forme
        fig.update_layout(
            title=f'Surface de Volatilité Interactive - {date.strftime("%Y-%m-%d")}<br>'
                  f'Spot: ${surface["metrics"]["spot_price"]:.2f} | '
                  f'ATM Vol: {surface["metrics"]["atm_vol"]:.1f}%',
            scene=dict(
                xaxis_title='Strike (K)',
                yaxis_title='Time to Maturity (années)',
                zaxis_title='Volatilité Implicite (%)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=700,
            width=900
        )
        
        fig.show()
        return fig
    
    def create_surface_evolution_gif(self, output_filename='volatility_evolution.gif', 
                                   frame_duration=800, max_frames=20):
        """
        Crée un GIF montrant l'évolution des surfaces de volatilité
        
        Args:
            output_filename: Nom du fichier de sortie
            frame_duration: Durée de chaque frame en ms
            max_frames: Nombre maximum de frames
        """
        if not self.volatility_surfaces:
            print("❌ Aucune surface disponible")
            return
        
        import matplotlib.animation as animation
        
        dates = sorted(list(self.volatility_surfaces.keys()))
        
        # Limiter le nombre de frames
        if len(dates) > max_frames:
            step = len(dates) // max_frames
            dates = dates[::step]
        
        print(f"🎬 Création du GIF avec {len(dates)} frames...")
        
        # Trouver les limites globales pour la cohérence
        all_strikes = []
        all_ttms = []
        all_vols = []
        
        for date in dates:
            surface = self.volatility_surfaces[date]
            all_strikes.extend(surface['clean_strikes'])
            all_ttms.extend(surface['clean_ttms'])
            all_vols.extend(surface['clean_vols'])
        
        strike_lim = [min(all_strikes), max(all_strikes)]
        ttm_lim = [min(all_ttms), max(all_ttms)]
        vol_lim = [min(all_vols), max(all_vols)]
        
        # Créer l'animation
        fig = plt.figure(figsize=(12, 8))
        
        def animate(frame):
            fig.clear()
            ax = fig.add_subplot(111, projection='3d')
            
            date = dates[frame]
            surface = self.volatility_surfaces[date]
            
            # Surface
            surf = ax.plot_surface(
                surface['strike_mesh'], 
                surface['ttm_mesh'], 
                surface['vol_mesh'],
                cmap='viridis', 
                alpha=0.8,
                vmin=vol_lim[0],
                vmax=vol_lim[1]
            )
            
            # Points réels
            ax.scatter(
                surface['clean_strikes'], 
                surface['clean_ttms'], 
                surface['clean_vols'],
                c='red', 
                s=15, 
                alpha=0.8
            )
            
            # Limites fixes
            ax.set_xlim(strike_lim)
            ax.set_ylim(ttm_lim)
            ax.set_zlim(vol_lim)
            
            # Labels
            ax.set_xlabel('Strike (K)')
            ax.set_ylabel('TTM (années)')
            ax.set_zlabel('Vol Implicite (%)')
            ax.set_title(f'Surface de Volatilité - {date.strftime("%Y-%m-%d")}\n'
                        f'ATM Vol: {surface["metrics"]["atm_vol"]:.1f}%')
        
        # Créer l'animation
        anim = animation.FuncAnimation(
            fig, animate, frames=len(dates), 
            interval=frame_duration, repeat=True
        )
        
        # Sauvegarder
        anim.save(output_filename, writer='pillow', fps=1000/frame_duration)
        print(f"🎬 GIF sauvegardé: {output_filename}")
        
        plt.close()
    
    def analyze_surface_metrics(self, start_date=None, end_date=None):
        """
        Analyse l'évolution des métriques de surface dans le temps
        
        Args:
            start_date: Date de début d'analyse
            end_date: Date de fin d'analyse
            
        Returns:
            DataFrame avec les métriques temporelles
        """
        if not self.volatility_surfaces:
            print("❌ Aucune surface disponible")
            return pd.DataFrame()
        
        print("📈 Analyse des métriques de surface...")
        
        # Extraire les métriques
        metrics_data = []
        
        for date, surface in self.volatility_surfaces.items():
            if start_date and date < pd.to_datetime(start_date):
                continue
            if end_date and date > pd.to_datetime(end_date):
                continue
                
            metrics = surface['metrics']
            metrics_data.append({
                'Date': date,
                'ATM_Vol': metrics['atm_vol'],
                'Vol_Min': metrics['vol_range'][0],
                'Vol_Max': metrics['vol_range'][1],
                'Vol_Std': metrics['vol_std'],
                'Skew': metrics.get('skew', np.nan),
                'Term_Structure': metrics.get('term_structure', np.nan),
                'Spot_Price': metrics['spot_price'],
                'N_Points': surface['n_points']
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        df_metrics.set_index('Date', inplace=True)
        df_metrics.sort_index(inplace=True)
        
        # Afficher les statistiques
        print(f"✅ Analyse terminée ({len(df_metrics)} dates):")
        print(f"   - ATM Vol moyenne: {df_metrics['ATM_Vol'].mean():.1f}%")
        print(f"   - Skew moyen: {df_metrics['Skew'].mean():.2f}%")
        print(f"   - Volatilité de volatilité: {df_metrics['ATM_Vol'].std():.1f}%")
        
        return df_metrics

# Fonction d'utilisation principale
def process_options_data(options_file_path, create_all_surfaces=True, 
                        interactive_date=None, create_gif=False):
    """
    Fonction principale pour traiter les données d'options et créer les surfaces
    
    Args:
        options_file_path: Chemin vers le fichier de données d'options
        create_all_surfaces: Créer toutes les surfaces quotidiennes
        interactive_date: Date pour visualisation interactive (format 'YYYY-MM-DD')
        create_gif: Créer un GIF d'évolution
    """
    print("🚀 Démarrage du traitement des données d'options")
    print("=" * 60)
    
    # Charger les données
    try:
        if options_file_path.endswith('.csv'):
            raw_data = pd.read_csv(options_file_path)
        else:
            raise ValueError("Format de fichier non supporté")
        print(f"✅ Données chargées: {len(raw_data)} lignes")
    except Exception as e:
        print(f"❌ Erreur chargement: {e}")
        return
    
    # Initialiser le transformateur
    transformer = OptionsDataTransformer()
    
    # Transformer les données
    structured_data = transformer.transform_to_structured_format(raw_data)
    
    if create_all_surfaces:
        # Créer toutes les surfaces
        surfaces = transformer.create_all_daily_surfaces()
        print(f"📊 {len(surfaces)} surfaces créées")
        
        # Analyser les métriques
        metrics_df = transformer.analyze_surface_metrics()
        
        # Sauvegarder les métriques
        metrics_df.to_csv('surface_metrics_analysis.csv')
        print("💾 Métriques sauvegardées: surface_metrics_analysis.csv")
    
    # Visualisation interactive si demandée
    if interactive_date:
        try:
            date_obj = pd.to_datetime(interactive_date)
            if create_all_surfaces or date_obj in transformer.volatility_surfaces:
                if date_obj not in transformer.volatility_surfaces:
                    # Créer juste cette surface
                    transformer.volatility_surfaces[date_obj] = transformer.create_daily_volatility_surface(date_obj)
                
                transformer.visualize_surface_interactive(date_obj)
                transformer.visualize_surface_3d(date_obj)
            else:
                print(f"❌ Date {interactive_date} non disponible")
        except Exception as e:
            print(f"❌ Erreur visualisation: {e}")
    
    # Créer le GIF si demandé
    if create_gif and len(transformer.volatility_surfaces) > 1:
        transformer.create_surface_evolution_gif()
    
    print("\n🎉 Traitement terminé!")
    return transformer

# Exemple d'utilisation
if __name__ == "__main__":
    # Remplacez par le chemin vers votre fichier de données
    file_path = "historical_options_data/AAPL_historical_options_6m_daily.csv"
    
    # Traitement complet
    transformer = process_options_data(
        options_file_path=file_path,
        create_all_surfaces=True,
        interactive_date="2025-05-27",  # Date pour visualisation
        create_gif=True
    )
