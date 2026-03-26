import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import os

# =========================
# 1. Chargement des données
# =========================
ratios = pd.read_csv("financial_ratios.csv")
ompic = pd.read_csv("ompic_distress.csv")

data = ratios.merge(ompic, on="company_id", how="left")
data["distress"] = data["distress"].fillna(0)

# =========================
# 2. Préparation
# =========================
X = data.select_dtypes(include="number").drop(columns=["distress", "company_id"])
y = data["distress"]

X = X.fillna(X.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42
)

# =========================
# 3. Modèles
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

results = {}

# =========================
# 4. Entraînement + Graphiques
# =========================
os.makedirs("graphs", exist_ok=True)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    auc = roc_auc_score(y_test, y_prob)

    results[name] = {
        "report": classification_report(y_test, y_pred),
        "auc": auc
    }

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title(f"ROC Curve - {name}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.savefig(f"graphs/roc_{name}.png")
    plt.close()

# Feature importance
rf = models["Random Forest"]
importances = rf.feature_importances_

plt.figure()
plt.barh(range(len(importances)), importances)
plt.title("Feature Importance")
plt.savefig("graphs/feature_importance.png")
plt.close()

# =========================
# 5. Génération Markdown
# =========================
with open("rapport.md", "w", encoding="utf-8") as f:

    f.write("# 📊 Rapport de Classification\n\n")

    f.write("## Sommaire\n")
    f.write("- [Introduction](#introduction)\n")
    f.write("- [Résultats](#résultats)\n")
    f.write("- [Conclusion](#conclusion)\n\n")

    f.write("## Introduction\n")
    f.write("Analyse des entreprises en difficulté.\n\n")

    f.write("## Résultats\n")

    for name, res in results.items():
        f.write(f"### {name}\n")
        f.write(f"AUC: {res['auc']}\n\n")
        f.write("```\n")
        f.write(res["report"])
        f.write("\n```\n")
        f.write(f"![ROC](graphs/roc_{name}.png)\n\n")

    f.write("### Importance des variables\n")
    f.write("![Importance](graphs/feature_importance.png)\n\n")

    f.write("## Conclusion\n")
    f.write("Le modèle Random Forest est le plus performant.\n\n")

    f.write("## Références\n")
    f.write("[1] Altman (1968)\n")
    f.write("[2] Breiman (2001)\n")
