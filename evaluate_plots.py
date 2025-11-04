# evaluate_plots.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
import joblib

def plot_confusion_matrix(y_true, y_pred, labels=None):
    cm = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

def plot_feature_importances(model, feature_names, top_n=20):
    # model must expose feature_importances_
    import numpy as np
    fi = model.named_steps['clf'].feature_importances_
    # If features are transformed by preprocessor (ohe), get names from preprocessor if possible.
    # This is a simplified show — adapt for your pipeline.
    indices = np.argsort(fi)[-top_n:]
    plt.barh(range(len(indices)), fi[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title("Top feature importances")
    plt.show()
