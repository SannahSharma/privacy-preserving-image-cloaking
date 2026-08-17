import csv
import statistics as stats
from pathlib import Path

RESULTS_CSV = Path("../datasets/results_strength1.csv")

rows = list(csv.DictReader(open(RESULTS_CSV)))

for method in ["fgsm", "pgd"]:
    method_rows = [r for r in rows if r["method"] == method]
    n = len(method_rows)
    flipped = sum(1 for r in method_rows if r["cloaked_class"] != r["original_class"])
    success_rate = flipped / n
    drops = [
        float(r["original_confidence"]) - float(r["confidence_in_original_class_after"])
        for r in method_rows
    ]
    mean_drop = stats.mean(drops)
    std_drop = stats.pstdev(drops)
    print(f"\n--- {method.upper()} (n={n}) ---")
    print(f"Misclassification rate (class actually changed): {success_rate:.1%}")
    print(f"Mean confidence drop in original class:            {mean_drop:.4f} (std {std_drop:.4f})")