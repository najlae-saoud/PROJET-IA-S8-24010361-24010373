# 📊 Rapport de Classification : Prédiction des Entreprises en Difficulté

> **Date :** Mars 2026  
> **Contexte :** Projet S8 IA - Encadrement et accompagnement en restructuration  
> **Datasets :** `financial_ratios.csv` (Kaggle), `ompic_distress.csv` (Données OMPIC)

---

## Table des matières

1. [Contexte et Objectifs](#1-contexte-et-objectifs)
2. [Préparation des données](#2-préparation-des-données)
3. [Régression Logistique (Baseline)](#3-régression-logistique-baseline)
4. [Forêts Aléatoires (Random Forest)](#4-forêts-aléatoires-random-forest)
5. [Évaluation et Résultats](#5-évaluation-et-résultats)
6. [Conclusion et Recommandations](#6-conclusion-et-recommandations)
7. [Références](#7-références)

---

## 1. Contexte et Objectifs

Ce projet vise à modéliser et identifier les entreprises nécessitant une **restructuration financière** de manière proactive. L'identification précoce des signaux de détresse financière (difficulté de trésorerie, baisse de rentabilité, fardeau de la dette) est primordiale pour proposer un accompagnement ciblé (via des institutions comme l'OMPIC) avant la cessation de paiement.

**Problématique :** Comment utiliser les ratios financiers historiques pour prédire si une entreprise va entrer en situation de détresse (variable cible binaire : `distress` = 1) ou rester saine (0) ?

---

## 2. Préparation des données

### 2.1 Sources et Fusion

Nous utilisons deux sources principales :
1. **Données Financières (Kaggle) :** Contient les variables prédictives (ratios de liquidité, rentabilité, levier financier, activité).
2. **Données OMPIC :** Contient la variable cible, indiquant le statut juridique ou de détresse de l'entreprise.

Une jointure à gauche (`LEFT JOIN`) est effectuée sur la clé `company_id`. Les valeurs manquantes de la cible `distress` sont imputées à `0` (entreprise supposée saine par défaut si elle n'est pas répertoriée en difficulté).

### 2.2 Nettoyage et Normalisation

Avant l'entraînement, les étapes suivantes sont indispensables pour garantir l'efficacité des modèles :

- **Sélection des caractéristiques numériques :** Les colonnes textuelles (ID, noms) sont écartées.
- **Imputation des valeurs manquantes :** Remplacement par la **médiane** de chaque ratio (plus robuste que la moyenne face aux valeurs extrêmes).
- **Mise à l'échelle (`StandardScaler`) :** Les ratios ayant des échelles très différentes (ex: marges en %, dettes en millions), nous les normalisons pour avoir une moyenne de 0 et une variance de 1.
- **Séparation Train / Test :** 70% pour l'entraînement, 30% pour l'évaluation finale.

---

## 3. Régression Logistique (Baseline)

### Principe Théorique
La régression logistique est un modèle statistique modélisant la probabilité qu'une entreprise appartienne à la classe "en difficulté" via une fonction logistique (sigmoïde). Il cherche la combinaison linéaire des ratios financiers qui sépare le mieux les deux classes.

```math
P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + ... + \beta_n X_n)}}
```

**Avantages :** 
- Très interprétable (les coefficients $\beta$ indiquent l'importance et le sens de la relation).
- Rapide et simple, sert d'excellente base de référence (baseline).

**Limites :** 
- Suppose une relation linéaire entre les ratios log-odds et les probabilités, ce qui est souvent faux dans des structures financières complexes.

---

## 4. Forêts Aléatoires (Random Forest)

### Principe Théorique
Une Forêt Aléatoire est une méthode d'apprentissage **ensembliste** qui combine les prédictions de multiples **arbres de décision** indépendants. Chaque arbre est entraîné sur un sous-échantillon aléatoire (Bootstrap) des données et un sous-ensemble aléatoire de caractéristiques à chaque nœud (Feature subsetting).

**Avantages pour l'analyse financière :**
- Capture les **relations non linéaires** complexes et les interactions entre les ratios (ex: une forte dette n'est dangereuse que si la liquidité est faible).
- Extrêmement robuste aux valeurs aberrantes de ratios financiers (très fréquentes).
- Permet d'extraire naturellement l'**importance des variables** (Feature Importance).

### Paramètres clés utilisés :
- `n_estimators = 200` : Nombre d'arbres utilisés, offre une grande stabilité.
- `random_state = 42` : Garantit la reproductibilité.

---

## 5. Évaluation et Résultats

Les modèles sont évalués sur les métriques clés suivantes :
- **Précision (Precision) :** Sur toutes les entreprises prédites "en difficulté", combien le sont réellement ? (Minimise les faux positifs).
- **Sensibilité (Recall) :** Sur toutes les entreprises réellement en difficulté, combien avons-nous réussi à détecter ? (Minimise les faux négatifs).
- **AUC-ROC :** Capacité globale du modèle à distinguer les deux classes, indépendamment du seuil de décision (Score de 0.5 = aléatoire, 1.0 = parfait).

### Comparaison des performances attendues

*Note: En l'absence de l'exécution sur les données réelles au moment de l'écriture du rapport, voici les tendances typiquement observées :*

| Modèle | Avantages observés | Interprétation |
|--------|----------------|----------------|
| **Régression Logistique** | Bonne calibration des probabilités | Peine si les frontières de décision sont non linéaires. |
| **Random Forest** | Excellente sensitivité (Recall), AUC souvent > 0.85 | Capable de détecter des profils complexes de faillite. |

### Importance des Variables (Feature Importance)
L'algorithme *Random Forest* identifie généralement les types de variables suivants dans le top 10 des facteurs de restructuration :
1. **Ratios de Liquidité** (Current Ratio, Quick Ratio)
2. **Profitabilité** (ROA, ROE, Marges Nettes)
3. **Levier Financier** (Debt-to-Equity Ratio, Ratio de couverture des intérêts)
Ces métriques valident empiriquement les théories économiques de la défaillance (ex: modèle Z-Score d'Altman).

_(Les graphiques `roc_curve.png` et `feature_importance.png` générés par le script illustrent visuellement ces métriques)._

---

## 6. Conclusion et Recommandations

Le modèle **Forêts Aléatoires (Random Forest)** se révèle généralement le plus performant pour ce type de problématique grâce à sa capacité à contourner la linéarité stricte. 

**Recommandations Métiers :**
1. **Intégration OMPIC :** Ce modèle prédictif peut être utilisé comme un **système d'alerte précoce** (Early Warning System) par les auditeurs financiers et l'OMPIC.
2. **Politique d'Accompagnement :** Les entreprises identifiées avec une forte probabilité (ex: > 70%) devraient se voir proposer un diagnostic de restructuration en amont.
3. **Suivi Temporel :** Il serait pertinent dans une V2 d'intégrer l'évolution temporelle des ratios (séries temporelles) plutôt qu'une vision statique, afin de capter la "vitesse" de dégradation de la santé financière.

---

## 7. Références

1. **Financial Ratios Dataset** - Kaggle ([Lien / Source originale]).
2. **Données OMPIC** - Registre sur les statuts légaux et difficultés des entreprises marocaines.
3. Documentation Officielle `scikit-learn` (RandomForestClassifier, LogisticRegression).
4. Altman, E. I. (1968). *Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy*. The Journal of Finance, 23(4), 589-609.
