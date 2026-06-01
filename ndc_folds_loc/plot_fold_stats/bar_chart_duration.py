import pandas as pd
import matplotlib.pyplot as plt

sex = pd.read_csv("/home/idatro/dialect_project/ndc_folds_loc/fold_stats_csv/sex_summary.csv")

# optionally remove exceptions if you only want actual folds
sex_plot = sex[sex["label"].str.startswith("fold_")].copy()

pivot = sex_plot.pivot(index="label", columns="group_value", values="hours_pct")

pivot.plot(kind="bar", stacked=True, figsize=(8, 5))
plt.ylabel("Duration share (%)")
plt.xlabel("Fold")
plt.title("Sex distribution by duration share across folds")
plt.legend(title="Sex")
plt.tight_layout()
plt.savefig("/home/idatro/dialect_project/ndc_folds_loc/plot_fold_stats/sex_bar_chart.pdf", bbox_inches="tight")
plt.show()