# 📊 Guide Complet du Pricing d'Options et Autocall

## Table des Matières

1. [🎯 Introduction - Pourquoi ce Projet?](#introduction)
2. [🏗️ Architecture du Projet](#architecture)
3. [📚 Concepts Fondamentaux](#concepts-fondamentaux)
4. [🛤️ Parcours d'Apprentissage](#parcours-apprentissage)
5. [📊 Structure des Données](#structure-données)
6. [🚀 Guide de Démarrage Rapide](#démarrage-rapide)
7. [📈 Analyse des Corrélations](#correlations)
8. [⚖️ Avantages et Désavantages](#avantages-desavantages)
9. [🔧 Guide Technique Détaillé](#guide-technique)
10. [📝 Exemples et Cas d'Usage](#exemples)

---

## 🎯 Introduction - Pourquoi ce Projet? {#introduction}

Ce projet vous permet de comprendre et maîtriser le pricing d'options et d'autocall à travers une approche pratique et théorique. Imaginez que vous êtes un trader qui doit:

- **Évaluer** le juste prix d'une option à tout moment
- **Comprendre** comment la volatilité influence les prix
- **Prédire** l'évolution des prix dans différents scénarios
- **Gérer** le risque de portefeuilles complexes

### 🎓 Ce que vous allez apprendre

1. **La théorie** : Modèles de Black-Scholes, surfaces de volatilité, Greeks
2. **La pratique** : Collecte de données réelles, calculs de prix, simulations
3. **L'intuition** : Comprendre viscéralement comment les paramètres interagissent
4. **L'application** : Pricer des produits exotiques comme les autocall

---

## 🏗️ Architecture du Projet {#architecture}

```
options-pricing-platform/
│
├── 📊 data_collection/        # Collecte des données de marché
│   ├── polygon_collector.py   # API Polygon.io (données professionnelles)
│   ├── yahoo_collector.py     # Yahoo Finance (données gratuites)
│   └── data_validator.py      # Validation et nettoyage
│
├── 🔄 data_processing/        # Transformation et enrichissement
│   ├── surface_builder.py     # Construction des surfaces de volatilité
│   ├── greeks_calculator.py   # Calcul des sensibilités
│   └── data_transformer.py    # Format structuré pour ML
│
├── 🤖 ml_models/             # Modèles de machine learning
│   ├── volatility_predictor.py # Prédiction de volatilité
│   ├── price_calibrator.py    # Calibration des modèles
│   └── surface_interpolator.py # Interpolation avancée
│
├── 🎲 simulation/            # Monte Carlo et backtesting
│   ├── monte_carlo_engine.py  # Moteur de simulation
│   ├── path_generator.py      # Génération de trajectoires
│   └── risk_calculator.py     # Métriques de risque
│
├── 📈 volatility_surface/    # Visualisation et analyse
│   ├── surface_visualizer.py  # Graphiques 3D interactifs
│   ├── metrics_analyzer.py    # Analyse des métriques
│   └── evolution_tracker.py   # Suivi temporel
│
├── 📁 data/                  # Données (ignoré par git)
├── 📊 results/               # Résultats et graphiques
├── 📚 docs/                  # Documentation détaillée
├── 🧪 tests/                 # Tests unitaires
├── requirements.txt          # Dépendances Python
└── README.md                 # Ce fichier
```

---

## 📚 Concepts Fondamentaux {#concepts-fondamentaux}

### 1. **Qu'est-ce qu'une Option?**

Une option est un **contrat** qui donne le droit (mais pas l'obligation) d'acheter ou vendre un actif à un prix fixé à l'avance.

**Analogie du quotidien** : C'est comme réserver une table au restaurant. Vous payez une petite somme pour avoir le *droit* de dîner à 20h, mais vous n'êtes pas *obligé* de venir.

#### Types d'options :
- **Call** : Droit d'ACHETER (vous pariez sur la hausse)
- **Put** : Droit de VENDRE (vous pariez sur la baisse)

### 2. **Les Paramètres Clés (et leurs corrélations)**

| Paramètre | Symbole | Description | Impact sur le Prix |
|-----------|---------|-------------|-------------------|
| **Spot** | S | Prix actuel de l'actif | ↑S → ↑Call, ↓Put |
| **Strike** | K | Prix d'exercice | ↑K → ↓Call, ↑Put |
| **Temps** | T | Durée jusqu'à l'échéance | ↑T → ↑Prix (généralement) |
| **Volatilité** | σ | Incertitude du marché | ↑σ → ↑Prix (toujours) |
| **Taux** | r | Taux sans risque | ↑r → ↑Call, ↓Put |

### 3. **Les Greeks - Les Sensibilités**

Les Greeks mesurent comment le prix de l'option change quand un paramètre bouge:

- **Delta (Δ)** : Sensibilité au prix du sous-jacent
  - Call: 0 à 1 (se comporte comme 0 à 100% de l'action)
  - Put: -1 à 0
  
- **Gamma (Γ)** : Accélération du Delta
  - Maximum à la monnaie (ATM)
  - Mesure le risque de couverture

- **Theta (Θ)** : Décroissance temporelle
  - Toujours négatif pour l'acheteur
  - S'accélère près de l'échéance

- **Vega (ν)** : Sensibilité à la volatilité
  - Maximum pour les options ATM
  - Plus important pour les longues maturités

### 4. **La Surface de Volatilité**

La volatilité n'est PAS constante! Elle varie selon:
- **Le Strike** : Phénomène du "smile" de volatilité
- **La Maturité** : Structure par terme
- **Le Temps** : Évolution dynamique

**Pourquoi c'est crucial?** Car la volatilité est le SEUL paramètre non observable directement!

### 5. **Qu'est-ce qu'un Autocall?**

Un autocall est un produit structuré qui:
- Se rembourse automatiquement si le sous-jacent dépasse un niveau prédéfini
- Paie des coupons conditionnels
- Offre une protection partielle du capital

**Analogie** : C'est comme un placement à terme avec des sorties anticipées bonus.

---

## 📊 Structure des Données {#structure-données}

### Format Principal des Données

```python
# Structure de la table principale enrichie
{
    'dataDate': '2025-05-28',           # Date de la donnée
    'surfaceId': 'surface_vol_1',       # Identifiant de la surface
    'ticker': 'AAPL',                   # Symbole du sous-jacent
    'spotPrice': 150.25,                # Prix du sous-jacent
    'optionType': 'CALL',               # Type d'option
    'strike': 165.0,                    # Prix d'exercice
    'moneyness': '110%',                # K/S ratio
    'expirationDate': '2025-11-28',     # Date d'échéance
    'timeToMaturity': 0.5,              # Temps en années
    'optionPrice': 3.89,                # Prix de l'option
    'impliedVolatility': 25.0,          # Volatilité implicite (%)
    'delta': 0.35,                      # Sensibilité au spot
    'gamma': 0.015,                     # Sensibilité du delta
    'theta': -0.08,                     # Décroissance temporelle
    'vega': 0.12,                       # Sensibilité à la vol
    'volume': 1250,                     # Volume échangé
    'openInterest': 5420,               # Positions ouvertes
    'bidAskSpread': 0.05                # Spread de marché
}
```

### Filtres Appliqués

- **Maturités** : 7 jours ≤ T ≤ 730 jours (2 ans)
- **Moneyness** : 70% ≤ K/S ≤ 130%
- **Liquidité** : Volume > 10 ou OpenInterest > 50
- **Volatilité** : 5% ≤ σ ≤ 200%

---

## 🚀 Guide de Démarrage Rapide {#démarrage-rapide}

### 1. Installation

```bash
# Cloner le repository
git clone https://github.com/awkwy/myriam.git
cd myriam

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copier le fichier de configuration
cp config.example.yaml config.yaml

# Éditer avec vos clés API
# - Polygon.io : Pour données professionnelles
# - Yahoo Finance : Données gratuites (limitées)
```

### 3. Premier Lancement

```bash
# Collecter les données (Yahoo Finance gratuit)
python data_collection/yahoo_collector.py --ticker AAPL --days 180

# Construire les surfaces de volatilité
python data_processing/surface_builder.py --input data/AAPL_options.csv

# Visualiser
python volatility_surface/surface_visualizer.py --date 2025-05-28
```

---

## 📈 Analyse des Corrélations {#correlations}

### Matrice des Corrélations Principales

| Relation | Corrélation | Explication | Implication Trading |
|----------|-------------|-------------|-------------------|
| **Spot ↔ Vol Implicite** | Négative (-0.3 à -0.7) | Quand le marché baisse, la peur augmente | Acheter des puts OTM comme assurance |
| **Temps ↔ Theta** | Non-linéaire | Accélération près de l'échéance | Éviter les options court terme sauf stratégie spécifique |
| **Vol Implicite ↔ Vol Réalisée** | ~0.6 | La vol implicite surestime souvent | Opportunité de vendre de la volatilité |
| **Moneyness ↔ Volume** | En U | Plus de volume ATM et deep OTM | Meilleure liquidité à ces niveaux |

### Phénomènes Observables

1. **Smile de Volatilité**
   - Les options OTM ont une vol implicite plus élevée
   - Plus prononcé en période de stress
   - Asymétrique (plus marqué côté put)

2. **Term Structure**
   - Court terme : Vol élevée avant événements
   - Long terme : Retour à la moyenne
   - Inversion possible en cas de stress

3. **Sticky Strike vs Sticky Delta**
   - Comment la surface bouge avec le spot
   - Important pour le risk management
   - Varie selon les marchés

---

## ⚖️ Avantages et Désavantages {#avantages-desavantages}

### ✅ Avantages de cette Approche

1. **Données Réelles**
   - Utilisation de vraies données de marché
   - Validation des modèles en temps réel
   - Apprentissage des conditions actuelles

2. **Modularité**
   - Chaque composant est indépendant
   - Facile d'ajouter de nouvelles fonctionnalités
   - Tests unitaires possibles

3. **Pédagogie**
   - Code commenté et expliqué
   - Progression graduelle
   - Exemples concrets

4. **Production-Ready**
   - Architecture scalable
   - Gestion des erreurs
   - Logging et monitoring

### ❌ Limitations et Défis

1. **Modèle Black-Scholes**
   - Hypothèses irréalistes (vol constante)
   - Pas de prise en compte des dividendes (sauf ajustement)
   - Assume des marchés parfaits

2. **Données de Marché**
   - Coût des données temps réel
   - Qualité variable selon les sources
   - Biais de survie pour l'historique

3. **Complexité Computationnelle**
   - Monte Carlo coûteux pour les exotiques
   - Calibration non-triviale
   - Trade-off précision vs vitesse

4. **Risques de Modèle**
   - Dépendance aux hypothèses
   - Régimes de marché changeants
   - Events extrêmes mal capturés

---

## 🔧 Guide Technique Détaillé {#guide-technique}

### 1. Collecte de Données (`data_collection/`)

#### `polygon_collector.py` - Données Professionnelles
```python
# Points clés:
- Rate limiting intelligent
- Gestion des erreurs avec retry
- Cache pour optimiser les requêtes
- Filtrage par liquidité et maturité
```

**Avantages Polygon.io:**
- Données tick par tick
- Historique complet
- API robuste

**Inconvénients:**
- Payant ($$$)
- Limite de requêtes
- Complexité accrue

#### `yahoo_collector.py` - Alternative Gratuite
```python
# Points clés:
- Pas de limite de requêtes
- Données end-of-day
- Facile à implémenter
```

**Quand utiliser Yahoo:**
- Prototypage rapide
- Projets éducatifs
- Budget limité

### 2. Construction des Surfaces (`data_processing/`)

#### Interpolation Methods Comparées

| Méthode | Avantages | Inconvénients | Cas d'Usage |
|---------|-----------|---------------|-------------|
| **Linear** | Rapide, stable | Peu smooth | Données denses |
| **Cubic** | Bon compromis | Oscillations possibles | Standard |
| **RBF** | Très smooth | Lent, overfitting | Présentation |
| **SVI** | Sans arbitrage | Complexe à calibrer | Production |

### 3. Machine Learning (`ml_models/`)

#### Architecture du Modèle de Volatilité
```
Input Layer (7 features)
    ↓
Dense Layer (128 neurons, ReLU)
    ↓
Dropout (0.2)
    ↓
Dense Layer (64 neurons, ReLU)
    ↓
Dense Layer (32 neurons, ReLU)
    ↓
Output Layer (1 neuron, Sigmoid scaled)
```

**Features utilisées:**
1. Moneyness (K/S)
2. Time to maturity
3. Historical volatility (20d)
4. Volume ratio
5. Put/Call ratio
6. Term structure slope
7. Skew level

### 4. Simulation Monte Carlo (`simulation/`)

#### Optimisations Implémentées

1. **Vectorisation NumPy**
   - 100x plus rapide que les boucles
   - Utilisation de la mémoire optimisée

2. **Variance Reduction**
   - Antithetic variates
   - Control variates
   - Importance sampling

3. **Parallélisation**
   - Multi-threading pour les paths
   - GPU support (optionnel)

### 5. Pricing d'Autocall

#### Décomposition du Produit
```
Autocall = Zero Coupon Bond 
         + Série de Call Spreads digitaux (pour les coupons)
         + Down-and-In Put (pour la protection)
         - Knock-out feature (complexité)
```

#### Algorithme de Pricing
1. Générer N trajectoires de prix
2. Pour chaque trajectoire:
   - Vérifier les conditions d'autocall à chaque date
   - Si déclenché: calculer le payoff
   - Sinon: continuer jusqu'à maturité
3. Moyenne des payoffs actualisés

---

## 📝 Exemples et Cas d'Usage {#exemples}

### Exemple 1: Pricer une Option Simple

```python
from data_processing.greeks_calculator import price_option

# Paramètres
S = 100  # Spot
K = 105  # Strike
T = 0.25  # 3 mois
r = 0.05  # Taux
sigma = 0.20  # Volatilité 20%

# Calcul
call_price = price_option('call', S, K, T, r, sigma)
print(f"Prix du Call: ${call_price:.2f}")
# Output: Prix du Call: $2.54
```

### Exemple 2: Analyser une Surface de Volatilité

```python
from volatility_surface.surface_visualizer import plot_surface

# Charger les données
surface_data = load_surface_data('AAPL', '2025-05-28')

# Visualiser
fig = plot_surface(surface_data)
fig.show()

# Extraire des métriques
atm_vol = surface_data.get_atm_volatility()
skew = surface_data.get_25d_skew()
term_structure = surface_data.get_term_structure()
```

### Exemple 3: Backtester une Stratégie

```python
from simulation.strategy_backtester import backtest_strategy

# Définir la stratégie
def my_strategy(market_data):
    # Acheter des calls 3 mois ATM quand vol < 15%
    if market_data['atm_vol_3m'] < 15:
        return {'action': 'buy', 'option': 'ATM_CALL_3M'}
    return {'action': 'hold'}

# Backtester
results = backtest_strategy(
    strategy=my_strategy,
    start_date='2024-01-01',
    end_date='2024-12-31',
    initial_capital=100000
)

print(f"Sharpe Ratio: {results['sharpe']:.2f}")
print(f"Max Drawdown: {results['max_dd']:.1f}%")
```

---

## 🎯 Projets Suggérés

### Niveau Débutant
1. **Calculateur de Greeks en temps réel**
2. **Visualiseur de stratégies simples** (straddle, butterfly)
3. **Simulateur de P&L** pour différents scénarios

### Niveau Intermédiaire
1. **Arbitrage detector** entre options
2. **Vol surface smoother** avec contraintes no-arbitrage
3. **Strategy optimizer** basé sur les Greeks

### Niveau Expert
1. **Market maker automatique** pour options
2. **Exotic pricer** (barriers, asiatiques, etc.)
3. **Risk management system** complet

---

## 📚 Ressources pour Approfondir

### Livres Essentiels
1. **"Options, Futures, and Other Derivatives"** - John Hull
   - La bible du domaine
   - Théorie + pratique

2. **"The Volatility Surface"** - Jim Gatheral
   - Focus sur la modélisation de la vol
   - Approche praticien

3. **"Dynamic Hedging"** - Nassim Taleb
   - Vision trader
   - Gestion des risques réels

### Papers Académiques Clés
1. **Black-Scholes (1973)** - Le modèle fondateur
2. **Heston (1993)** - Volatilité stochastique
3. **SVI (2004)** - Surfaces sans arbitrage

### Ressources en Ligne
- **QuantLib** : Librairie C++ de référence
- **Wilmott Forums** : Communauté de quants
- **arXiv Quantitative Finance** : Papers récents

---

## 🤝 Contribution

Ce projet est conçu pour être étendu! Voici comment contribuer:

1. **Fork** le repository
2. **Créez** une branche (`git checkout -b feature/AmazingFeature`)
3. **Committez** vos changements (`git commit -m 'Add AmazingFeature'`)
4. **Push** vers la branche (`git push origin feature/AmazingFeature`)
5. **Ouvrez** une Pull Request

### Idées de Contributions
- Nouveaux modèles de pricing
- Sources de données additionnelles
- Optimisations de performance
- Documentation dans d'autres langues
- Cas d'études réels

---

## 📜 License

Ce projet est sous license MIT. Voir le fichier `LICENSE` pour plus de détails.

---

*"The market can stay irrational longer than you can stay solvent."* - John Maynard Keynes

Bon trading et bon apprentissage! 🚀
