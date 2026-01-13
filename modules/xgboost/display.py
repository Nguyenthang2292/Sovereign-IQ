"""
Display and reporting functions for xgboost_prediction_main.py
"""

from colorama import Fore
from sklearn.metrics import classification_report, confusion_matrix

from config import TARGET_LABELS
from modules.common.utils import log_analysis, log_info, log_model

from .utils import color_text


def print_classification_report(y_true, y_pred, title="Classification Report"):
    """
    Prints a formatted classification report with color coding.
    """
    print()
    log_analysis("=" * 60)
    log_analysis(title)
    log_analysis("=" * 60)

    # Get classification report as string
    report = classification_report(
        y_true,
        y_pred,
        target_names=TARGET_LABELS,
        output_dict=False,
    )
    print(report)

    # Print confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    log_model("Confusion Matrix:")
    log_info("(Rows = True, Columns = Predicted)")
    print(" " * 12, end="")
    for label in TARGET_LABELS:
        print(f"{label:>12}", end="")
    print()
    for i, label in enumerate(TARGET_LABELS):
        print(f"{label:>12}", end="")
        for j in range(len(TARGET_LABELS)):
            value = cm[i, j]
            # Color code: green for correct predictions (diagonal), red for major errors
            if i == j:
                color = Fore.GREEN
            elif abs(i - j) == 2:  # UP vs DOWN or vice versa
                color = Fore.RED
            else:
                color = Fore.YELLOW
            print(color_text(f"{value:>12}", color), end="")
        print()

    log_analysis("=" * 60)
    print()
