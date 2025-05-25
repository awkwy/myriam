# Projet : Analyse et Prédiction des Prix d'Options avec des Modèles Machine Learning

## 📌 Vue d'ensemble

Ce projet développe une approche innovante pour modéliser la surface de volatilité implicite et prédire les prix d'options financières sur les actions Apple (AAPL) en utilisant des techniques avancées de Machine Learning. L'objectif est de dépasser les limites des modèles traditionnels comme Black-Scholes en capturant les dynamiques non-linéaires complexes du marché des options.

## 🎯 Objectifs du Projet

1. **Modélisation de la surface de volatilité implicite** : Créer une représentation tridimensionnelle précise de la volatilité implicite en fonction du strike et de la maturité.

2. **Prédiction des prix d'options** : Développer des modèles prédictifs robustes capables d'estimer avec précision les prix des options calls et puts.

3. **Analyse comparative des modèles** : Identifier le modèle ML optimal en termes de précision, robustesse et efficacité computationnelle.

## 📊 Données Utilisées et Leur Pertinence

### Variables Principales

#### 1. Prix du Sous-jacent (Spot Price - S)
**Pertinence** : Variable fondamentale car elle détermine la valeur intrinsèque de l'option. La relation entre le prix spot et le strike définit si l'option est dans la monnaie (ITM), à la monnaie (ATM) ou hors de la monnaie (OTM).

**Impact** : Les variations du prix spot affectent directement le delta de l'option et sa probabilité d'exercice.

#### 2. Prix d'Exercice (Strike - K)
**Pertinence** : Détermine le niveau auquel l'option peut être exercée. Le ratio S/K (moneyness) est crucial pour comprendre le comportement de l'option.

**Impact** : Influence directement la valeur intrinsèque et la sensibilité de l'option aux mouvements du sous-jacent.

#### 3. Temps jusqu'à Maturité (Time to Maturity - T)
**Pertinence** : La valeur temps représente une composante majeure du prix de l'option. Plus l'échéance est lointaine, plus l'option a de chances de devenir profitable.

**Impact** : Affecte le theta (décroissance temporelle) et la probabilité que l'option finisse ITM.

#### 4. Volatilité Implicite (σ_implied)
**Pertinence** : Représente les anticipations du marché concernant la volatilité future. C'est le paramètre le plus sensible et le plus difficile à estimer.

**Impact** : Une augmentation de 1% de la volatilité peut significativement augmenter le prix de l'option, particulièrement pour les options ATM.

#### 5. Taux Sans Risque (r)
**Pertinence** : Représente le coût d'opportunité du capital et influence la valeur présente des flux futurs.

**Impact** : Affecte principalement le rho de l'option et devient plus significatif pour les options de longue maturité.

### Variables Potentiellement Manquantes et Leur Importance

#### 1. Dividendes (q)
**Importance critique** : Les dividendes réduisent le prix du sous-jacent à la date ex-dividende, impactant significativement les prix des options, surtout pour les puts.

**Recommandation** : Intégrer le rendement en dividendes attendu ou les dates/montants de dividendes discrets.

#### 2. Greeks de Second Ordre
- **Gamma** : Mesure la convexité du delta. Crucial pour comprendre les risques de couverture.
- **Vanna** : Sensibilité du delta à la volatilité. Important pour les stratégies de volatilité.
- **Volga** : Sensibilité du vega à la volatilité. Essentiel pour le risk management.

#### 3. Volatilité Réalisée Historique
**Importance** : Permet de comparer la volatilité implicite avec la volatilité historique pour identifier les opportunités d'arbitrage.

**Utilisation** : Calculer sur différentes fenêtres (10, 20, 30, 60 jours) pour capturer différents régimes de marché.

#### 4. Volume et Open Interest
**Importance** : Indicateurs de liquidité essentiels pour évaluer la qualité des prix et les coûts de transaction.

**Impact** : Les options peu liquides peuvent avoir des spreads bid-ask importants affectant la rentabilité.

#### 5. Skew de Volatilité
**Importance** : Capture l'asymétrie du smile de volatilité, particulièrement prononcée pour les indices.

**Calcul** : Différence de volatilité implicite entre puts OTM et calls OTM de même distance au strike ATM.

#### 6. Structure à Terme de la Volatilité
**Importance** : Les volatilités implicites varient selon les maturités, créant une structure à terme.

**Application** : Permet de mieux prédire les prix d'options de différentes échéances.

#### 7. Indicateurs de Marché
- **VIX** : Indice de volatilité du marché, proxy pour le sentiment de risque global.
- **Corrélations inter-marchés** : Relations avec d'autres actifs (indices, commodités, devises).

## 🌐 Surface de Volatilité Implicite

La surface de volatilité est une représentation tridimensionnelle montrant comment la volatilité implicite varie en fonction du strike et de la maturité. Cette visualisation révèle plusieurs phénomènes importants :

### Caractéristiques Observées

1. **Smile de Volatilité** : Les options OTM (particulièrement les puts) présentent souvent des volatilités implicites plus élevées, reflétant la demande de protection contre les baisses.

2. **Structure à Terme** : La volatilité tend à converger vers une moyenne à long terme (mean reversion).

3. **Dynamiques Asymétriques** : Les surfaces ne sont pas symétriques, avec des pentes différentes côté put et call.

### Défis et Limites

- **Liquidité Variable** : Les zones avec peu de transactions créent des discontinuités artificielles.
- **Extrapolation aux Extrêmes** : La précision diminue pour les strikes très éloignés ou les maturités très courtes/longues.
- **Arbitrage de Calendrier** : Des incohérences peuvent apparaître entre différentes maturités.

## ⚙️ Modèles Machine Learning Évalués

### 1. Régression Linéaire & Ridge

**Concept** : Modèles de base utilisant des relations linéaires entre variables.

**Avantages** :
- Interprétabilité directe des coefficients
- Rapidité d'entraînement et de prédiction
- Ridge ajoute une régularisation L2 réduisant le surapprentissage

**Limites** :
- Incapacité à capturer les non-linéarités inhérentes aux options
- Performance médiocre sur les relations complexes entre strike, maturité et volatilité

**Cas d'usage optimal** : Benchmark initial ou options très proches de l'ATM avec peu de temps jusqu'à maturité.

### 2. Random Forest

**Concept** : Ensemble d'arbres de décision votant pour la prédiction finale.

**Avantages** :
- Capture naturellement les interactions non-linéaires
- Robuste au bruit et aux outliers
- Fournit l'importance des variables
- Peu de risque de surapprentissage grâce au bagging

**Limites** :
- Peut être lent sur de très grands datasets
- Difficulté à extrapoler au-delà des valeurs d'entraînement
- Consommation mémoire importante

**Cas d'usage optimal** : Données avec beaucoup de bruit ou quand l'interprétabilité des features est importante.

### 3. Gradient Boosting

**Concept** : Construction séquentielle d'arbres corrigeant les erreurs des précédents.

**Avantages** :
- Excellente précision prédictive
- Gestion efficace des interactions complexes
- Ajustement fin possible via learning rate

**Limites** :
- Sensible au surapprentissage sans régularisation appropriée
- Plus lent que XGBoost ou LightGBM
- Nécessite un tuning minutieux des hyperparamètres

**Cas d'usage optimal** : Quand la précision est prioritaire et que le temps d'entraînement n'est pas critique.

### 4. XGBoost (eXtreme Gradient Boosting) - ⭐ Meilleur Modèle

**Concept** : Version optimisée du gradient boosting avec régularisation avancée.

**Avantages** :
- Performance prédictive exceptionnelle (R² proche de 1)
- Régularisation L1/L2 intégrée
- Gestion native des valeurs manquantes
- Parallélisation efficace
- Pruning d'arbres intelligent

**Optimisations clés** :
```python
params = {
    'max_depth': 6-10,           # Profondeur optimale pour options
    'learning_rate': 0.05-0.1,   # Balance vitesse/précision
    'n_estimators': 500-1000,    # Nombre d'arbres
    'subsample': 0.8,            # Échantillonnage pour robustesse
    'colsample_bytree': 0.8,     # Features par arbre
    'gamma': 0.1,                # Régularisation minimale par split
    'reg_alpha': 0.1,            # L1 regularization
    'reg_lambda': 1.0            # L2 regularization
}
```

**Limites** :
- Complexité computationnelle pour le tuning
- Risque de surapprentissage sur petits datasets
- Moins interprétable que les modèles linéaires

### 5. LightGBM

**Concept** : Gradient boosting optimisé pour la vitesse et l'efficacité mémoire.

**Avantages** :
- Extrêmement rapide (10x plus que XGBoost sur grands datasets)
- Consommation mémoire réduite
- Gestion efficace des features catégorielles
- Excellent pour les données haute dimension

**Innovations techniques** :
- Gradient-based One-Side Sampling (GOSS)
- Exclusive Feature Bundling (EFB)
- Histogram-based algorithm

**Limites** :
- Plus sensible au bruit que Random Forest
- Peut surapprendre sur petits datasets
- Nécessite des données bien préparées

### 6. Réseaux de Neurones (Multi-Layer Perceptron)

**Concept** : Modèles profonds capables d'approximer toute fonction continue.

**Architecture typique** :
```python
model = Sequential([
    Dense(128, activation='relu', input_dim=n_features),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')  # Prix de l'option
])
```

**Avantages** :
- Capacité théorique illimitée de modélisation
- Excellent pour capturer des patterns très complexes
- Peut intégrer différents types de données

**Limites** :
- Nécessite beaucoup de données pour bien généraliser
- Temps d'entraînement long
- Difficile à interpréter (boîte noire)
- Sensible à l'initialisation et à l'architecture

## 📈 Analyse des Performances par Type d'Option

### Options CALL
Les modèles performent généralement mieux sur les calls pour plusieurs raisons :
- Patterns de volatilité plus stables
- Moins d'asymétrie dans la distribution des rendements
- Demande plus prévisible basée sur les anticipations de hausse

### Options PUT
Les puts présentent des défis supplémentaires :
- Skew de volatilité plus prononcé (fear gauge)
- Demande spike durant les périodes de stress
- Nécessité de capturer les queues de distribution (tail risk)

**Recommandation** : Entraîner des modèles séparés pour calls et puts peut améliorer significativement les performances.

## 🔄 Stratégie de Validation et Robustesse

### Validation Croisée Temporelle
Pour les données financières, une validation croisée standard n'est pas appropriée. Utilisez plutôt :

```python
# Walk-forward validation
for train_end in monthly_dates:
    train = data[data.date < train_end]
    test = data[(data.date >= train_end) & 
                (data.date < train_end + 1_month)]
    model.fit(train)
    evaluate(model, test)
```

### Métriques d'Évaluation
- **R² Score** : Variance expliquée (cible : > 0.95)
- **MAPE** : Erreur en pourcentage (cible : < 5%)
- **MAE en $** : Erreur absolue moyenne en dollars
- **Quantile Loss** : Pour évaluer les queues de distribution

## 💡 Recommandations pour l'Amélioration

### 1. Enrichissement des Features
- Ajouter les Greeks calculés numériquement
- Intégrer des indicateurs techniques du sous-jacent
- Inclure des variables macroéconomiques (taux, inflation)

### 2. Approches Avancées
- **Modèles Ensemble** : Combiner XGBoost avec des réseaux de neurones
- **Transfer Learning** : Utiliser des modèles pré-entraînés sur d'autres actifs
- **Modèles Stochastiques** : Intégrer des processus de volatilité stochastique

### 3. Gestion du Risque
- Implémenter des contraintes d'absence d'arbitrage
- Vérifier la cohérence des surfaces de volatilité générées
- Backtesting sur différents régimes de marché

## 📂 Structure du Projet

```
project/
├── data/
│   ├── raw/              # Données brutes de marché
│   ├── processed/        # Features engineered
│   └── surfaces/         # Surfaces de volatilité
├── models/
│   ├── baseline/         # Modèles simples
│   ├── ensemble/         # Modèles avancés
│   └── saved/           # Modèles entraînés
├── src/
│   ├── finance3.py      # Pipeline principal
│   ├── pranav.py        # Collecte de données
│   ├── features.py      # Engineering des features
│   └── evaluation.py    # Métriques et validation
└── results/
    ├── model_comparison_results.csv
    └── visualizations/
```

## 🖥️ Installation et Exécution

### Prérequis
```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### Installation des Dépendances
```bash
pip install -r requirements.txt
```

### Configuration
```python
# config.py
TICKER = 'AAPL'
START_DATE = '2023-01-01'
END_DATE = '2024-12-31'
RISK_FREE_RATE = 0.05  # Taux T-Bill 3 mois
```

### Exécution
```bash
# Collecter les données
python pranav.py

# Lancer l'analyse complète
python finance3.py

# Générer les visualisations
python visualize_results.py
```

## 📊 Résultats Attendus

1. **Surfaces de volatilité** : Visualisations 3D interactives
2. **Comparaison des modèles** : Tableau détaillé des performances
3. **Prédictions** : Prix d'options avec intervalles de confiance
4. **Feature Importance** : Analyse de l'importance des variables

## 📖 Ressources et Références

### Livres Fondamentaux
- Hull, J. "Options, Futures, and Other Derivatives" - Bible des produits dérivés
- Wilmott, P. "Paul Wilmott on Quantitative Finance" - Approche mathématique approfondie
- Gatheral, J. "The Volatility Surface" - Focus sur la modélisation de volatilité

### Articles Académiques
- Hutchinson et al. (1994) "A Nonparametric Approach to Pricing and Hedging Derivative Securities"
- Culkin & Das (2017) "Machine Learning in Finance: The Case of Deep Learning for Option Pricing"

### Documentation Technique
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

---

📧 **Contact** : Pour questions ou contributions, ouvrez une issue sur le repository.

🌟 **Note** : Ce projet est à but éducatif et de recherche. Les modèles ne constituent pas des conseils d'investissement.
