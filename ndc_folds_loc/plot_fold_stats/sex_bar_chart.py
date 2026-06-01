
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sex = pd.read_csv("/home/idatro/dialect_project/ndc_folds_loc/fold_stats_csv/sex_summary.csv")
sex_plot = sex[sex["label"].str.startswith("fold_")].copy()

plt.figure(figsize=(8, 5))
sns.barplot(data=sex_plot, x="label", y="total_hours", hue="group_value")
#change name of legend to "sex"

plt.legend(title="Sex", title_fontsize=14, fontsize=14)
#change name of x-axis ticks to "fold 1", "fold 2", etc.
plt.xticks(ticks=range(len(sex_plot["label"].unique())), labels=[f"{i+1}" for i in range(len(sex_plot["label"].unique()))], size=14)
plt.yticks(fontsize=14)
plt.ylim(0, sex_plot["total_hours"].max() * 1.3)  # Add some space above the tallest bar
plt.ylabel("Total hours", fontsize=16)
plt.xlabel("Fold", fontsize=16)
plt.title("Total duration per sex across folds", fontsize=18)
plt.tight_layout()
plt.savefig("/home/idatro/dialect_project/ndc_folds_loc/plot_fold_stats/sex_bar_chart.pdf", bbox_inches="tight")
plt.show()