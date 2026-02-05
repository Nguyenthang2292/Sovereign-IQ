"""
Display and reporting functions for xgboost_prediction_main.py
"""

import numpy as np
from colorama import Fore
from sklearn.metrics import classification_report, confusion_matrix

from config import TARGET_LABELS
from modules.common.utils import log_analysis, log_info, log_model
from modules.xgboost.utils.utils import color_text


def print_classification_report(y_true, y_pred, title="Classification Report"):
    """
    Prints a formatted classification report with color coding.
    When the test set contains only a subset of classes (e.g. 2 of 3),
    uses labels/target_names for the classes present to avoid sklearn ValueError.
    """
    print()
    log_analysis("=" * 60)
    log_analysis(title)
    log_analysis("=" * 60)

    # Use only labels that appear in y_true or y_pred (avoids "2 classes vs 3 target_names" error)
    labels_present = np.array(
        sorted(set(np.unique(y_true)) | set(np.unique(y_pred))), dtype=np.intp
    )
    target_names_present = [TARGET_LABELS[i] for i in labels_present]

    report = classification_report(
        y_true,
        y_pred,
        labels=labels_present,
        target_names=target_names_present,
        output_dict=False,
    )
    print(report)

    # Confusion matrix with same label order
    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    log_model("Confusion Matrix:")
    log_info("(Rows = True, Columns = Predicted)")
    print(" " * 12, end="")
    for name in target_names_present:
        print(f"{name:>12}", end="")
    print()
    for i, name in enumerate(target_names_present):
        print(f"{name:>12}", end="")
        for j in range(len(labels_present)):
            value = cm[i, j]
            if i == j:
                color = Fore.GREEN
            elif abs(labels_present[i] - labels_present[j]) == 2:
                color = Fore.RED
            else:
                color = Fore.YELLOW
            print(color_text(f"{value:>12}", color), end="")
        print()

    log_analysis("=" * 60)
    print()
