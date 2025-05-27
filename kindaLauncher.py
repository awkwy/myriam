"""
Script d'exemple pour utiliser le transformateur de données d'options
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Supposons que vous ayez vos données dans un fichier CSV
# Vous pouvez utiliser soit les données de finance0.py soit de finance5.py

def example_with_current_data():
    """
    Exemple avec vos données actuelles (format finance0.py)
    """
    print("📊 Exemple avec données actuelles")
    
    # Charger vos données d'options existantes
    # Remplacez par le bon chemin de fichier
    options_file = "options_data/AAPL_options_data_ml_ready.csv"
    
    try:
        df = pd.read_csv(options_file)
        print(f"✅ Données chargées: {len(df)} options")
        
        # Vérifier les colonnes disponibles
        print(f"Colonnes disponibles: {list(df.columns)}")
        
        # Si vous n'avez qu'une seule date, on peut simuler plusieurs dates
        if 'Date_Scraping' in df.columns:
            # Ajouter des dates fictives pour créer une série temporelle
            base_date = pd.to_datetime(df['Date_Scraping'].iloc[0])
            df['dataDate'] = base_date  # Utiliser comme date de base
            
            # Créer des variations pour simuler plusieurs jours
            n_days = 5  # Simuler 5 jours
            all_data = []
            
            for i in range(n_days):
                df_day = df.copy()
                df_day['dataDate'] = base_date - timedelta(days=i)
                
                # Simuler des variations de prix et volatilité
                price_var = 1 + np.random.normal(0, 0.02, len(df_day))  # ±2% variation
                vol_var = 1 + np.random.normal(0, 0.05, len(df_day))    # ±5% variation vol
                
                df_day['lastPrice'] = df_day['Prix_Option'] * price_var
                df_day['impliedVolatilityPct'] = df_day['Volatilite_Implicite'] * vol_var
                
                all_data.append(df_day)
            
            # Combiner toutes les dates
            df_multi = pd.concat(all_data, ignore_index=True)
            
        else:
            df_multi = df.copy()
            df_multi['dataDate'] = datetime.now().strftime('%Y-%m-%d')
        
        # Maintenant utiliser le transformateur
        transformer = OptionsDataTransformer()
        
        # Mapping des colonnes selon votre format
        if 'Prix_Option' in df_multi.columns:
            price_col = 'Prix_Option'
        elif 'lastPrice' in df_multi.columns:
            price_col = 'lastPrice'
        else:
            price_col = 'Prix_Option'  # Par défaut
        
        if 'Volatilite_Implicite' in df_multi.columns:
            vol_col = 'Volatilite_Implicite'
        elif 'impliedVolatilityPct' in df_multi.columns:
            vol_col = 'impliedVolatilityPct'
        else:
            vol_col = 'impliedVolatilityPct'
        
        # Transformer en format structuré
        structured_data = transformer.transform_to_structured_format(
            df_multi, 
            price_col=price_col,
            vol_col=vol_col,
            date_col='dataDate'
        )
        
        print("\n📋 Aperçu des données structurées:")
        print(structured_data.head(10))
        
        # Créer une surface pour la première date disponible
        first_date = structured_data.index.get_level_values(0)[0]
        print(f"\n🎯 Création de surface pour {first_date}")
        
        surface = transformer.create_daily_volatility_surface(first_date)
        
        if surface:
            print("✅ Surface créée avec succès!")
            
            # Visualisation 3D
            transformer.volatility_surfaces[first_date] = surface
            fig = transformer.visualize_surface_3d(first_date)
            
            # Afficher les métriques
            metrics = surface['metrics']
            print(f"\n📊 Métriques de la surface:")
            print(f"   - Prix spot: ${metrics['spot_price']:.2f}")
            print(f"   - Vol ATM: {metrics['atm_vol']:.1f}%")
            print(f"   - Skew: {metrics.get('skew', 'N/A')}")
            print(f"   - Points utilisés: {surface['n_points']}")
            
        else:
            print("❌ Impossible de créer la surface")
        
        return transformer
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None

def example_with_historical_data():
    """
    Exemple avec données historiques (format finance5.py)
    """
    print("📊 Exemple avec données historiques")
    
    # Si vous avez utilisé finance5.py pour générer des données historiques
    historical_file = "historical_options_data/AAPL_historical_options_6m_daily.csv"
    
    try:
        df = pd.read_csv(historical_file)
        print(f"✅ Données historiques chargées: {len(df)} options")
        
        transformer = OptionsDataTransformer()
        
        # Transformer les données
        structured_data = transformer.transform_to_structured_format(
            df,
            price_col='lastPrice',
            vol_col='impliedVolatility',  # Attention: ici c'est en décimal, pas en %
            date_col='dataDate'
        )
        
        # Créer toutes les surfaces
        surfaces = transformer.create_all_daily_surfaces()
        
        if surfaces:
            print(f"✅ {len(surfaces)} surfaces créées!")
            
            # Analyser les métriques
            metrics_df = transformer.analyze_surface_metrics()
            print("\n📈 Évolution des métriques:")
            print(metrics_df.head())
            
            # Visualisation de la dernière date
            last_date = max(surfaces.keys())
            transformer.visualize_surface_interactive(last_date)
            
            # Créer un GIF d'évolution (si plusieurs dates)
            if len(surfaces) > 3:
                transformer.create_surface_evolution_gif()
        
        return transformer
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None

def create_sample_data():
    """
    Crée des données d'exemple si vous n'avez pas encore de fichier
    """
    print("🔧 Création de données d'exemple...")
    
    np.random.seed(42)  # Pour reproductibilité
    
    # Paramètres de base
    spot_price = 150
    dates = pd.date_range(start='2025-05-20', end='2025-05-27', freq='D')
    strikes = np.arange(120, 181, 5)  # Strikes de 120 à 180 par pas de 5
    expirations = ['2025-06-20', '2025-07-18', '2025-08-15']  # 3 échéances
    
    data = []
    
    for date in dates:
        for exp in expirations:
            exp_date = pd.to_datetime(exp)
            ttm = (exp_date - date).days / 365.25
            
            for strike in strikes:
                moneyness = strike / spot_price
                
                # Simuler volatilité avec smile
                base_vol = 0.25  # 25% vol de base
                smile_effect = 0.1 * abs(np.log(moneyness)) ** 1.5  # Effet smile
                vol_noise = np.random.normal(0, 0.02)  # Bruit
                
                impl_vol = (base_vol + smile_effect + vol_noise) * 100  # En %
                
                # Prix d'option approximatif (simplifié)
                intrinsic = max(0, spot_price - strike)  # Call ITM
                time_value = impl_vol/100 * spot_price * np.sqrt(ttm) * 0.4
                option_price = intrinsic + time_value
                
                # Greeks approximatifs
                delta = 0.5 + 0.4 * np.tanh((spot_price - strike) / (0.2 * spot_price))
                gamma = 0.02 * np.exp(-0.5 * ((spot_price - strike) / (0.3 * spot_price))**2)
                theta = -option_price * 0.1 / 365  # Approximatif
                vega = spot_price * 0.01 * np.sqrt(ttm)
                
                # CALL
                data.append({
                    'dataDate': date.strftime('%Y-%m-%d'),
                    'optionType': 'CALL',
                    'strike': strike,
                    'expirationDate': exp,
                    'lastPrice': max(0.01, option_price),
                    'yearsToExpiration': ttm,
                    'impliedVolatilityPct': max(5, min(200, impl_vol)),
                    'underlyingPrice': spot_price,
                    'theta': theta,
                    'delta': delta,
                    'gamma': gamma,
                    'vega': vega
                })
                
                # PUT (approximatif avec put-call parity)
                put_price = option_price + strike - spot_price
                put_delta = delta - 1
                
                data.append({
                    'dataDate': date.strftime('%Y-%m-%d'),
                    'optionType': 'PUT',
                    'strike': strike,
                    'expirationDate': exp,
                    'lastPrice': max(0.01, put_price),
                    'yearsToExpiration': ttm,
                    'impliedVolatilityPct': max(5, min(200, impl_vol)),
                    'underlyingPrice': spot_price,
                    'theta': theta,
                    'delta': put_delta,
                    'gamma': gamma,
                    'vega': vega
                })
    
    df = pd.DataFrame(data)
    
    # Sauvegarder
    df.to_csv('sample_options_data.csv', index=False)
    print(f"✅ Données d'exemple créées: {len(df)} options")
    print(f"💾 Sauvegardées dans: sample_options_data.csv")
    
    return df

def main():
    """
    Fonction principale - choisissez votre scénario
    """
    print("🚀 Transformateur de données d'options - Exemples d'utilisation")
    print("=" * 70)
    
    # Scénario 1: Utiliser vos données existantes
    print("\n1️⃣  Test avec données actuelles...")
    transformer1 = example_with_current_data()
    
    # Scénario 2: Créer des données d'exemple si nécessaire
    print("\n2️⃣  Création de données d'exemple...")
    sample_df = create_sample_data()
    
    # Tester avec les données d'exemple
    transformer2 = OptionsDataTransformer()
    structured = transformer2.transform_to_structured_format(sample_df)
    
    # Créer quelques surfaces
    surfaces = transformer2.create_all_daily_surfaces()
    
    if surfaces:
        # Analyser les métriques
        metrics = transformer2.analyze_surface_metrics()
        print("\n📊 Métriques des surfaces d'exemple:")
        print(metrics)
        
        # Visualiser une surface
        first_date = list(surfaces.keys())[0]
        transformer2.visualize_surface_3d(first_date)
    
    print("\n🎉 Exemples terminés!")

if __name__ == "__main__":
    main()
