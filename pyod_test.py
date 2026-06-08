"""
PyOD Installationstest
Testet drei Anomalieerkennungs-Algorithmen auf synthetischen Daten.
Aufruf: python3 pyod_test.py
"""

import sys

# --- 1. Versionscheck --------------------------------------------------------
try:
    import pyod
    print(f"PyOD Version : {pyod.__version__}")
except ImportError:
    print("FEHLER: PyOD nicht installiert.")
    print("Installation: pip install pyod --break-system-packages")
    sys.exit(1)

import numpy as np
from pyod.models.knn     import KNN
from pyod.models.iforest import IForest
from pyod.models.lof     import LOF
from pyod.utils.data     import generate_data

# --- 2. Testdaten generieren -------------------------------------------------
# 200 Trainingspunkte, 100 Testpunkte, 2 Features, 10 % Anomalien
X_train, X_test, y_train, y_test = generate_data(
    n_train=200,
    n_test=100,
    n_features=2,
    contamination=0.1,
    random_state=42,
)
print(f"\nTrainingsdaten : {X_train.shape[0]} Punkte ({int(y_train.sum())} Anomalien)")
print(f"Testdaten      : {X_test.shape[0]} Punkte ({int(y_test.sum())} Anomalien)")
print(f"Features       : {X_train.shape[1]}")

# --- 3. Drei Algorithmen testen ----------------------------------------------
algorithmen = [
    ("KNN (k-Nearest Neighbors)", KNN()),
    ("Isolation Forest",          IForest(random_state=42)),
    ("LOF (Local Outlier Factor)",LOF()),
]

print("\n" + "-" * 55)
print(f"{'Algorithmus':<30} {'Anom':>4}  {'Prec':>6}  {'Recall':>6}  {'F1':>6}")
print("-" * 55)

alle_ok = True
for name, clf in algorithmen:
    clf.fit(X_train)
    y_pred = clf.predict(X_test)          # 0 = normal, 1 = Anomalie

    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    status = "OK" if f1 > 0.7 else "WARNUNG"
    if status != "OK":
        alle_ok = False

    print(f"{name:<30} {int(y_pred.sum()):>4}  {precision:>6.3f}  {recall:>6.3f}  {f1:>6.3f}  [{status}]")

print("-" * 55)

# --- 4. Ergebnis -------------------------------------------------------------
print()
if alle_ok:
    print("PyOD ist korrekt installiert und funktioniert.")
else:
    print("WARNUNG: Mindestens ein Algorithmus liefert unerwartete Ergebnisse.")

print(f"\nPython   : {sys.version.split()[0]}")
print(f"NumPy    : {np.__version__}")
print(f"PyOD     : {pyod.__version__}")
