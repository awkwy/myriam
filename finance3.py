"""
Modélisation de la Surface de Volatilité et Comparaison de Modèles ML
Pour la prédiction des prix d'options AAPL
Auteur: Assistant Claude
Date: Mai 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import xgboost as xgb
import lightgbm as lgb

# Pour les graphiques avancés
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =====================================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# =====================================================

print("📊 Chargement des données d'options...")
print("=" * 60)

# Charger les données (utiliser le fichier généré par le script précédent)
try:
    df = pd.read_csv('aapl_options_data_filter.csv')
    print(f"✅ {len(df)} options chargées")
except:
    print("⚠️  Fichier non trouvé. Génération de données simulées pour la démonstration...")
    # Génération de données simulées pour la démonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Paramètres de simulation
    spot_price = 150
    risk_free_rate = 0.05
    
    df = pd.DataFrame({
        'Strike': np.random.uniform(100, 200, n_samples),
        'Temps_Maturite_Annees': np.random.uniform(0.02, 3, n_samples),
        'Prix_Sous_Jacent': spot_price,
        'Volatilite_Implicite': np.random.uniform(15, 60, n_samples),
        'Type_Option': np.random.choice(['CALL', 'PUT'], n_samples),
        'Taux_Sans_Risque': risk_free_rate
    })
    
    # Calcul approximatif du prix d'option (Black-Scholes simplifié pour la démo)
    def approx_option_price(row):
        S = row['Prix_Sous_Jacent']
        K = row['Strike']
        T = row['Temps_Maturite_Annees']
        vol = row['Volatilite_Implicite'] / 100
        
        moneyness = S / K
        if row['Type_Option'] == 'CALL':
            intrinsic = max(0, S - K)
        else:
            intrinsic = max(0, K - S)
        
        time_value = S * vol * np.sqrt(T) * 0.4
        return intrinsic + time_value * np.random.uniform(0.8, 1.2)
    
    df['Prix_Option'] = df.apply(approx_option_price, axis=1)
    df['Moneyness'] = df['Strike'] / df['Prix_Sous_Jacent']
    df['Jours_Jusqu_Maturite'] = df['Temps_Maturite_Annees'] * 365

# =====================================================
# 2. CRÉATION DE LA SURFACE DE VOLATILITÉ 3D
# =====================================================

print("\n🎨 Création de la surface de volatilité...")

# Filtrer les données pour une meilleure visualisation
calls_df = df[df['Type_Option'] == 'CALL'].copy()

# Créer la figure 3D avec matplotlib
fig = plt.figure(figsize=(15, 10))

# Surface de volatilité pour les CALLS
ax1 = fig.add_subplot(121, projection='3d')

# Créer une grille pour l'interpolation
strike_range = np.linspace(calls_df['Strike'].min(), calls_df['Strike'].max(), 50)
maturity_range = np.linspace(calls_df['Temps_Maturite_Annees'].min(), 
                            calls_df['Temps_Maturite_Annees'].max(), 50)
strike_grid, maturity_grid = np.meshgrid(strike_range, maturity_range)

# Interpoler la volatilité implicite
points = calls_df[['Strike', 'Temps_Maturite_Annees']].values
values = calls_df['Volatilite_Implicite'].values
vol_grid = griddata(points, values, (strike_grid, maturity_grid), method='cubic')

# Créer la surface
surf = ax1.plot_surface(strike_grid, maturity_grid, vol_grid,
                       cmap=cm.viridis, alpha=0.8, linewidth=0, antialiased=True)

# Ajouter les points de données réels
ax1.scatter(calls_df['Strike'], calls_df['Temps_Maturite_Annees'], 
           calls_df['Volatilite_Implicite'], c='red', alpha=0.3, s=1)

ax1.set_xlabel('Strike Price ($)')
ax1.set_ylabel('Time to Maturity (Years)')
ax1.set_zlabel('Implied Volatility (%)')
ax1.set_title('Surface de Volatilité - CALLS')
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

# Surface de volatilité pour les PUTS
puts_df = df[df['Type_Option'] == 'PUT'].copy()
ax2 = fig.add_subplot(122, projection='3d')

if len(puts_df) > 0:
    points_puts = puts_df[['Strike', 'Temps_Maturite_Annees']].values
    values_puts = puts_df['Volatilite_Implicite'].values
    vol_grid_puts = griddata(points_puts, values_puts, (strike_grid, maturity_grid), method='cubic')
    
    surf2 = ax2.plot_surface(strike_grid, maturity_grid, vol_grid_puts,
                            cmap=cm.plasma, alpha=0.8, linewidth=0, antialiased=True)
    ax2.scatter(puts_df['Strike'], puts_df['Temps_Maturite_Annees'], 
               puts_df['Volatilite_Implicite'], c='blue', alpha=0.3, s=1)
    
    ax2.set_xlabel('Strike Price ($)')
    ax2.set_ylabel('Time to Maturity (Years)')
    ax2.set_zlabel('Implied Volatility (%)')
    ax2.set_title('Surface de Volatilité - PUTS')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

plt.tight_layout()
plt.savefig('volatility_surface_3d.png', dpi=300, bbox_inches='tight')
plt.show()

# =====================================================
# 3. SURFACE DE VOLATILITÉ INTERACTIVE AVEC PLOTLY
# =====================================================

print("\n🌐 Création de la surface de volatilité interactive...")

# Créer une surface interactive avec Plotly
fig_plotly = go.Figure()

# Surface pour les CALLS
fig_plotly.add_trace(go.Surface(
    x=strike_range,
    y=maturity_range,
    z=vol_grid,
    colorscale='Viridis',
    name='CALLS',
    showscale=True,
    colorbar=dict(title="Vol (%)", x=1.1)
))

# Points de données réels
fig_plotly.add_trace(go.Scatter3d(
    x=calls_df['Strike'],
    y=calls_df['Temps_Maturite_Annees'],
    z=calls_df['Volatilite_Implicite'],
    mode='markers',
    marker=dict(size=2, color='red', opacity=0.5),
    name='Données CALLS'
))

fig_plotly.update_layout(
    title='Surface de Volatilité Implicite Interactive - Options AAPL',
    scene=dict(
        xaxis_title='Strike Price ($)',
        yaxis_title='Time to Maturity (Years)',
        zaxis_title='Implied Volatility (%)',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    ),
    width=1000,
    height=800
)

fig_plotly.write_html('volatility_surface_interactive.html')
print("✅ Surface interactive sauvegardée dans 'volatility_surface_interactive.html'")

# =====================================================
# 4. PRÉPARATION DES DONNÉES POUR LE MACHINE LEARNING
# =====================================================

print("\n🤖 Préparation des données pour le Machine Learning...")

# Sélectionner les features pertinentes
features = ['Prix_Sous_Jacent', 'Strike', 'Volatilite_Implicite', 
           'Temps_Maturite_Annees', 'Taux_Sans_Risque', 'Moneyness']

# Créer des features supplémentaires
df['Strike_Ratio'] = df['Strike'] / df['Prix_Sous_Jacent']
df['Sqrt_Time'] = np.sqrt(df['Temps_Maturite_Annees'])
df['Vol_Time'] = df['Volatilite_Implicite'] * df['Sqrt_Time']
df['Log_Moneyness'] = np.log(df['Moneyness'])

# Ajouter les nouvelles features
extended_features = features + ['Strike_Ratio', 'Sqrt_Time', 'Vol_Time', 'Log_Moneyness']

# Encoder le type d'option
df['Is_Call'] = (df['Type_Option'] == 'CALL').astype(int)
extended_features.append('Is_Call')

# Préparer X et y
X = df[extended_features]
y = df['Prix_Option']

# Diviser les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation des features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Données préparées:")
print(f"   - Features: {len(extended_features)}")
print(f"   - Train set: {len(X_train)} samples")
print(f"   - Test set: {len(X_test)} samples")

# =====================================================
# 5. DÉFINITION ET ENTRAÎNEMENT DES MODÈLES
# =====================================================

print("\n🏋️ Entraînement des modèles de Machine Learning...")
print("=" * 60)

# Dictionnaire pour stocker les résultats
results = {}

# 1. RÉGRESSION LINÉAIRE (Baseline)
print("\n1️⃣ Régression Linéaire...")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
results['Linear Regression'] = {
    'model': lr,
    'predictions': y_pred_lr,
    'mse': mean_squared_error(y_test, y_pred_lr),
    'mae': mean_absolute_error(y_test, y_pred_lr),
    'r2': r2_score(y_test, y_pred_lr),
    'mape': mean_absolute_percentage_error(y_test, y_pred_lr) * 100
}

# 2. RIDGE REGRESSION
print("2️⃣ Ridge Regression...")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)
results['Ridge'] = {
    'model': ridge,
    'predictions': y_pred_ridge,
    'mse': mean_squared_error(y_test, y_pred_ridge),
    'mae': mean_absolute_error(y_test, y_pred_ridge),
    'r2': r2_score(y_test, y_pred_ridge),
    'mape': mean_absolute_percentage_error(y_test, y_pred_ridge) * 100
}

# 3. RANDOM FOREST
print("3️⃣ Random Forest...")
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)  # RF n'a pas besoin de normalisation
y_pred_rf = rf.predict(X_test)
results['Random Forest'] = {
    'model': rf,
    'predictions': y_pred_rf,
    'mse': mean_squared_error(y_test, y_pred_rf),
    'mae': mean_absolute_error(y_test, y_pred_rf),
    'r2': r2_score(y_test, y_pred_rf),
    'mape': mean_absolute_percentage_error(y_test, y_pred_rf) * 100
}

# 4. GRADIENT BOOSTING
print("4️⃣ Gradient Boosting...")
gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
results['Gradient Boosting'] = {
    'model': gb,
    'predictions': y_pred_gb,
    'mse': mean_squared_error(y_test, y_pred_gb),
    'mae': mean_absolute_error(y_test, y_pred_gb),
    'r2': r2_score(y_test, y_pred_gb),
    'mape': mean_absolute_percentage_error(y_test, y_pred_gb) * 100
}

# 5. XGBOOST
print("5️⃣ XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
results['XGBoost'] = {
    'model': xgb_model,
    'predictions': y_pred_xgb,
    'mse': mean_squared_error(y_test, y_pred_xgb),
    'mae': mean_absolute_error(y_test, y_pred_xgb),
    'r2': r2_score(y_test, y_pred_xgb),
    'mape': mean_absolute_percentage_error(y_test, y_pred_xgb) * 100
}

# 6. LIGHTGBM
print("6️⃣ LightGBM...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)
results['LightGBM'] = {
    'model': lgb_model,
    'predictions': y_pred_lgb,
    'mse': mean_squared_error(y_test, y_pred_lgb),
    'mae': mean_absolute_error(y_test, y_pred_lgb),
    'r2': r2_score(y_test, y_pred_lgb),
    'mape': mean_absolute_percentage_error(y_test, y_pred_lgb) * 100
}

# 7. SUPPORT VECTOR REGRESSION
print("7️⃣ Support Vector Regression...")
svr = SVR(kernel='rbf', C=100, gamma='scale')
svr.fit(X_train_scaled, y_train)
y_pred_svr = svr.predict(X_test_scaled)
results['SVR'] = {
    'model': svr,
    'predictions': y_pred_svr,
    'mse': mean_squared_error(y_test, y_pred_svr),
    'mae': mean_absolute_error(y_test, y_pred_svr),
    'r2': r2_score(y_test, y_pred_svr),
    'mape': mean_absolute_percentage_error(y_test, y_pred_svr) * 100
}

# 8. NEURAL NETWORK
print("8️⃣ Neural Network (MLP)...")
nn = MLPRegressor(
    hidden_layer_sizes=(100, 50, 25),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)
nn.fit(X_train_scaled, y_train)
y_pred_nn = nn.predict(X_test_scaled)
results['Neural Network'] = {
    'model': nn,
    'predictions': y_pred_nn,
    'mse': mean_squared_error(y_test, y_pred_nn),
    'mae': mean_absolute_error(y_test, y_pred_nn),
    'r2': r2_score(y_test, y_pred_nn),
    'mape': mean_absolute_percentage_error(y_test, y_pred_nn) * 100
}

# =====================================================
# 6. COMPARAISON DES MODÈLES
# =====================================================

print("\n📊 Comparaison des performances des modèles:")
print("=" * 80)

# Créer un DataFrame des résultats
comparison_df = pd.DataFrame({
    model: {
        'MSE': results[model]['mse'],
        'MAE': results[model]['mae'],
        'R²': results[model]['r2'],
        'MAPE (%)': results[model]['mape'],
        'RMSE': np.sqrt(results[model]['mse'])
    }
    for model in results
}).T

# Trier par R² décroissant
comparison_df = comparison_df.sort_values('R²', ascending=False)

# Afficher le tableau de comparaison
print("\n" + comparison_df.round(4).to_string())

# Identifier le meilleur modèle
best_model_name = comparison_df.index[0]
print(f"\n🏆 Meilleur modèle: {best_model_name}")
print(f"   - R² Score: {comparison_df.loc[best_model_name, 'R²']:.4f}")
print(f"   - MAPE: {comparison_df.loc[best_model_name, 'MAPE (%)']:.2f}%")

# =====================================================
# 7. VISUALISATION DES PERFORMANCES
# =====================================================

print("\n📈 Création des visualisations de performance...")

# Figure avec 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Comparaison des métriques
ax1 = axes[0, 0]
metrics = ['R²', 'MAPE (%)']
x = np.arange(len(comparison_df))
width = 0.35

for i, metric in enumerate(metrics):
    offset = width * (i - 0.5)
    if metric == 'R²':
        values = comparison_df[metric]
    else:
        values = comparison_df[metric] / 10  # Échelle pour la visualisation
    ax1.bar(x + offset, values, width, label=metric if metric == 'R²' else 'MAPE/10')

ax1.set_xlabel('Modèles')
ax1.set_ylabel('Score')
ax1.set_title('Comparaison des Métriques de Performance')
ax1.set_xticks(x)
ax1.set_xticklabels(comparison_df.index, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Scatter plot des prédictions vs valeurs réelles (meilleur modèle)
ax2 = axes[0, 1]
best_predictions = results[best_model_name]['predictions']
ax2.scatter(y_test, best_predictions, alpha=0.5, s=10)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_xlabel('Valeurs Réelles')
ax2.set_ylabel('Prédictions')
ax2.set_title(f'Prédictions vs Réalité - {best_model_name}')
ax2.grid(True, alpha=0.3)

# 3. Distribution des erreurs
ax3 = axes[1, 0]
errors = {}
for model_name in results:
    errors[model_name] = y_test.values - results[model_name]['predictions']

# Box plot des erreurs
ax3.boxplot([errors[model] for model in comparison_df.index], 
            labels=comparison_df.index)
ax3.set_xlabel('Modèles')
ax3.set_ylabel('Erreur de Prédiction ($)')
ax3.set_title('Distribution des Erreurs par Modèle')
ax3.set_xticklabels(comparison_df.index, rotation=45, ha='right')
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)

# 4. Importance des features (pour les modèles basés sur les arbres)
ax4 = axes[1, 1]
if best_model_name in ['Random Forest', 'XGBoost', 'LightGBM', 'Gradient Boosting']:
    model = results[best_model_name]['model']
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]  # Top 10 features
        
        ax4.bar(range(len(indices)), importances[indices])
        ax4.set_xlabel('Features')
        ax4.set_ylabel('Importance')
        ax4.set_title(f'Importance des Features - {best_model_name}')
        ax4.set_xticks(range(len(indices)))
        ax4.set_xticklabels([extended_features[i] for i in indices], rotation=45, ha='right')
    else:
        ax4.text(0.5, 0.5, 'Feature importance\nnon disponible\npour ce modèle', 
                ha='center', va='center', transform=ax4.transAxes)
else:
    ax4.text(0.5, 0.5, 'Feature importance\nnon disponible\npour ce modèle', 
            ha='center', va='center', transform=ax4.transAxes)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# =====================================================
# 8. ANALYSE DÉTAILLÉE PAR TYPE D'OPTION
# =====================================================

print("\n🔍 Analyse détaillée par type d'option...")

# Séparer les données de test par type
test_df = X_test.copy()
test_df['Prix_Reel'] = y_test
test_df['Type_Option'] = test_df['Is_Call'].map({1: 'CALL', 0: 'PUT'})

# Analyser les performances par type
for option_type in ['CALL', 'PUT']:
    mask = test_df['Type_Option'] == option_type
    if mask.sum() > 0:
        print(f"\n{option_type}S:")
        for model_name in comparison_df.index[:3]:  # Top 3 modèles
            preds = results[model_name]['predictions'][mask]
            real = test_df.loc[mask, 'Prix_Reel']
            mape = mean_absolute_percentage_error(real, preds) * 100
            print(f"  {model_name}: MAPE = {mape:.2f}%")

# =====================================================
# 9. VALIDATION CROISÉE DU MEILLEUR MODÈLE
# =====================================================

print("\n🔄 Validation croisée du meilleur modèle...")

best_model = results[best_model_name]['model']

# Préparer les données selon le type de modèle
if best_model_name in ['Linear Regression', 'Ridge', 'SVR', 'Neural Network']:
    X_cv = X_train_scaled
else:
    X_cv = X_train

# Validation croisée
cv_scores = cross_val_score(best_model.__class__(**best_model.get_params()), 
                           X_cv, y_train, cv=5, scoring='r2')

print(f"\nScores R² de validation croisée pour {best_model_name}:")
print(f"  Scores: {cv_scores.round(4)}")
print(f"  Moyenne: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# =====================================================
# 10. RECOMMANDATIONS ET CONCLUSIONS
# =====================================================

print("\n💡 Recommandations:")
print("=" * 60)

# Analyser les résultats
if comparison_df.iloc[0]['R²'] > 0.95:
    print("✅ Performance EXCELLENTE - Le modèle capture très bien la complexité des prix d'options")
elif comparison_df.iloc[0]['R²'] > 0.90:
    print("✅ Performance TRÈS BONNE - Le modèle est fiable pour la prédiction")
elif comparison_df.iloc[0]['R²'] > 0.80:
    print("⚠️  Performance CORRECTE - Des améliorations sont possibles")
else:
    print("❌ Performance INSUFFISANTE - Considérer d'autres approches")

print(f"\n🏆 Modèle recommandé: {best_model_name}")
print("\nAvantages de ce modèle:")

model_advantages = {
    'XGBoost': [
        "- Excellent équilibre entre performance et vitesse",
        "- Gestion native des valeurs manquantes",
        "- Régularisation intégrée contre le surapprentissage"
    ],
    'LightGBM': [
        "- Très rapide pour l'entraînement et la prédiction",
        "- Efficace en mémoire",
        "- Excellent pour les grandes bases de données"
    ],
    'Random Forest': [
        "- Robuste au bruit et aux outliers",
        "- Peu de risque de surapprentissage",
        "- Interprétabilité via l'importance des features"
    ],
    'Neural Network': [
        "- Capture les relations non-linéaires complexes",
        "- Adaptable à différents types de données",
        "- Performance potentiellement supérieure avec plus de données"
    ],
    'Gradient Boosting': [
        "- Haute précision de prédiction",
        "- Flexibilité dans la fonction de perte",
        "- Bonne gestion des interactions entre features"
    ]
}

if best_model_name in model_advantages:
    for advantage in model_advantages[best_model_name]:
        print(advantage)

print("\n📝 Suggestions d'amélioration:")
print("1. Ajouter plus de features (Greeks, données de marché)")
print("2. Utiliser des données de volatilité réalisée")
print("3. Implémenter un ensemble de modèles (stacking)")
print("4. Optimiser les hyperparamètres avec GridSearch")
print("5. Considérer des modèles spécifiques aux options (modèles de volatilité stochastique)")

# =====================================================
# 11. SAUVEGARDE DES RÉSULTATS
# =====================================================

print("\n💾 Sauvegarde des résultats...")

# Sauvegarder les métriques de comparaison
comparison_df.to_csv('model_comparison_results.csv')
print("✅ Résultats sauvegardés dans 'model_comparison_results.csv'")

# Sauvegarder le meilleur modèle
import joblib
joblib.dump(best_model, f'best_model_{best_model_name.replace(" ", "_")}.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
print(f"✅ Meilleur modèle sauvegardé dans 'best_model_{best_model_name.replace(' ', '_')}.pkl'")

print("\n✨ Analyse complète terminée!")
print("=" * 60)
