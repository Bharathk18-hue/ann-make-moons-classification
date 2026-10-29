"""
Development of an Artificial Neural Network (ANN) for Binary Classification
Using the make_moons Dataset from Scikit-learn

Run:  python main.py
Plots are saved in the "outputs" folder and also shown on screen.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
SHOW_PLOTS = True          # set to False to only save plots (no pop-up windows)
os.makedirs("outputs", exist_ok=True)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


def finish_plot(filename):
    """Save the current figure, then optionally show it."""
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", filename), dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


# ---------------------------------------------------------------
# 1. Dataset generation
# ---------------------------------------------------------------
print("=" * 60, "\n1. DATASET GENERATION\n" + "=" * 60)
X, y = make_moons(n_samples=1000, noise=0.2, random_state=SEED)

df = pd.DataFrame(X, columns=["feature_1", "feature_2"])
df["target"] = y
print(df.head())
print(f"\nNumber of samples : {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print("Class counts:\n", df["target"].value_counts())

plt.figure(figsize=(7, 5))
plt.scatter(X[y == 0, 0], X[y == 0, 1], c="tab:blue", edgecolor="k", label="Class 0")
plt.scatter(X[y == 1, 0], X[y == 1, 1], c="tab:orange", edgecolor="k", label="Class 1")
plt.title("make_moons Dataset (1000 samples, noise=0.2)")
plt.xlabel("Feature 1"); plt.ylabel("Feature 2"); plt.legend(); plt.grid(alpha=0.3)
finish_plot("01_dataset.png")

# ---------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------
print("=" * 60, "\n2. PREPROCESSING\n" + "=" * 60)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit ONLY on training data
X_test_scaled = scaler.transform(X_test)         # reuse the same scaler

print("Training set:", X_train.shape, " Testing set:", X_test.shape)
print("Scaled train mean (approx 0):", X_train_scaled.mean(axis=0).round(3))
print("Scaled train std  (approx 1):", X_train_scaled.std(axis=0).round(3))

# ---------------------------------------------------------------
# 3. Build the ANN
# ---------------------------------------------------------------
print("=" * 60, "\n3. BUILD THE ANN\n" + "=" * 60)
model = Sequential([
    Input(shape=(2,)),                 # 2 input features
    Dense(16, activation="relu"),      # hidden layer 1
    Dense(8, activation="relu"),       # hidden layer 2
    Dense(1, activation="sigmoid"),    # output: probability of class 1
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# ---------------------------------------------------------------
# 4. Train the model
# ---------------------------------------------------------------
print("=" * 60, "\n4. TRAINING\n" + "=" * 60)
early_stop = EarlyStopping(monitor="val_loss", patience=10,
                           restore_best_weights=True, verbose=1)

history = model.fit(X_train_scaled, y_train,
                    validation_split=0.2,
                    epochs=100, batch_size=32,
                    callbacks=[early_stop], verbose=1)
print(f"\nTraining stopped after {len(history.history['loss'])} epochs.")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history.history["accuracy"], label="Training accuracy")
ax[0].plot(history.history["val_accuracy"], label="Validation accuracy")
ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Accuracy")
ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(history.history["loss"], label="Training loss")
ax[1].plot(history.history["val_loss"], label="Validation loss")
ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Binary cross-entropy")
ax[1].legend(); ax[1].grid(alpha=0.3)
finish_plot("02_training_curves.png")

# ---------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------
print("=" * 60, "\n5. EVALUATION ON TEST DATA\n" + "=" * 60)
test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
y_prob = model.predict(X_test_scaled, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)

ann_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1-score": f1_score(y_test, y_pred),
}
print(f"Test loss    : {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
for name, value in ann_metrics.items():
    print(f"{name:<10}: {value:.4f}")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion matrix:\n", cm)
print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))

ConfusionMatrixDisplay(cm, display_labels=["Class 0", "Class 1"]).plot(cmap="Blues")
plt.title("ANN Confusion Matrix (Test Data)")
finish_plot("03_confusion_matrix.png")

# ---------------------------------------------------------------
# 6. Decision boundary
# ---------------------------------------------------------------
def plot_decision_boundary(predict_fn, title, filename):
    """predict_fn takes SCALED inputs and returns probabilities."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]             # original-scale grid points
    grid_scaled = scaler.transform(grid)             # SAME scaler as training
    probs = predict_fn(grid_scaled).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, probs, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=0.6)
    plt.colorbar(label="Predicted probability of Class 1")
    plt.contour(xx, yy, probs, levels=[0.5], colors="k", linewidths=2)   # boundary
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c="tab:blue", edgecolor="k", label="Class 0")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="tab:red", edgecolor="k", label="Class 1")
    plt.title(title); plt.xlabel("Feature 1"); plt.ylabel("Feature 2"); plt.legend()
    finish_plot(filename)


print("=" * 60, "\n6. DECISION BOUNDARY\n" + "=" * 60)
plot_decision_boundary(lambda g: model.predict(g, verbose=0).ravel(),
                       "ANN Decision Boundary (black line = 0.5 threshold)",
                       "04_ann_decision_boundary.png")

# ---------------------------------------------------------------
# 7. Predict new inputs
# ---------------------------------------------------------------
def predict_new_point(feature_1, feature_2):
    """Return (predicted_class, probability_of_class_1) for one new point."""
    point = np.array([[feature_1, feature_2]])
    point_scaled = scaler.transform(point)           # same fitted scaler
    prob = float(model.predict(point_scaled, verbose=0)[0][0])
    return int(prob >= 0.5), prob


print("=" * 60, "\n7. TESTING NEW INPUTS\n" + "=" * 60)
for f1, f2 in [(0.0, 0.5), (1.0, -0.5), (-1.0, 0.0), (2.0, 0.5)]:
    cls, prob = predict_new_point(f1, f2)
    print(f"Input ({f1:5.2f}, {f2:5.2f}) -> class {cls}, P(class 1) = {prob:.4f}")

# ---------------------------------------------------------------
# 8. Baseline: Logistic Regression
# ---------------------------------------------------------------
print("=" * 60, "\n8. BASELINE: LOGISTIC REGRESSION\n" + "=" * 60)
log_reg = LogisticRegression(random_state=SEED)
log_reg.fit(X_train_scaled, y_train)
lr_pred = log_reg.predict(X_test_scaled)

lr_metrics = {
    "Accuracy": accuracy_score(y_test, lr_pred),
    "Precision": precision_score(y_test, lr_pred),
    "Recall": recall_score(y_test, lr_pred),
    "F1-score": f1_score(y_test, lr_pred),
}
comparison = pd.DataFrame({"ANN": ann_metrics, "Logistic Regression": lr_metrics}).round(4)
print(comparison)

comparison.plot(kind="bar", figsize=(8, 5), edgecolor="k")
plt.title("ANN vs Logistic Regression (Test Data)")
plt.ylabel("Score"); plt.ylim(0, 1.05); plt.xticks(rotation=0); plt.grid(axis="y", alpha=0.3)
finish_plot("05_model_comparison.png")

plot_decision_boundary(lambda g: log_reg.predict_proba(g)[:, 1],
                       "Logistic Regression Decision Boundary (a straight line)",
                       "06_logreg_decision_boundary.png")

print("\nDone. Plots saved in the 'outputs' folder.")
