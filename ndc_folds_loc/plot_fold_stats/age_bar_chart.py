
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

age = pd.read_csv("/home/idatro/dialect_project/ndc_folds_loc/fold_stats_csv/age_group_summary.csv")
age_plot = age[age["label"].str.startswith("fold_")].copy()

plt.figure(figsize=(8, 5))
sns.barplot(data=age_plot, x="label", y="total_hours", hue="group_value")
#change name of legend to "age_group"
plt.legend(title="Age group", title_fontsize=14, fontsize=14)
#change name of x-axis ticks to "fold 1", "fold 2", etc.
plt.xticks(ticks=range(len(age_plot["label"].unique())), labels=[f"{i+1}" for i in range(len(age_plot["label"].unique()))], size=14)
plt.yticks(fontsize=14)
plt.ylim(0, age_plot["total_hours"].max() * 1.3)  # Add some space above the tallest bar
plt.ylabel("Total hours", fontsize=16)
plt.xlabel("Fold", fontsize=16)
plt.title("Total duration per age group across folds", fontsize=18)
plt.tight_layout()
plt.savefig("/home/idatro/dialect_project/ndc_folds_loc/plot_fold_stats/age_bar_chart.pdf", bbox_inches="tight")
plt.show()
