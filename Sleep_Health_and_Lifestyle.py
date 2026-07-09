import math
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

# Load dataset
df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
print(df.head())
stress_level = df["Stress Level"].dropna()#clean all empty parts

# Descriptive statistics
mean = stress_level.mean()
median = stress_level.median()
standart_deviation = stress_level.std()
variance = stress_level.var()
standart_error = standart_deviation / math.sqrt(len(stress_level))

print(f"mean: {mean:.2f}")
print(f"median: {median:.2f}")
print(f"standard deviation: {standart_deviation:.2f}")
print(f"variance: {variance:.2f}")
print(f"standard error: {standart_error:.2f}")

# Histogram
plt.figure(figsize=(10, 4))
plt.hist(stress_level, bins=10, color='orchid', edgecolor='black')
plt.title("Histogram of Stress Level")
plt.xlabel("Stress Level")
plt.ylabel("Frequency")
plt.figtext(0.5, -0.1, "Most stress levels are concentrated between 4 and 6, indicating moderate stress.", ha='center', fontsize=10)
plt.tight_layout()
plt.show()

# Boxplot
plt.figure(figsize=(6, 4))
sns.boxplot(y=stress_level, color='lightgreen')
plt.title("Stress Level - Boxplot")
plt.figtext(0.5, -0.1, "Stress levels are fairly symmetrically distributed with no outliers.", ha='center', fontsize=10)
plt.tight_layout()
plt.show()

# Outlier detection
Q1 = stress_level.quantile(0.25)
Q3 = stress_level.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = stress_level[(stress_level < lower_bound) | (stress_level > upper_bound)]

print(f"values for Identify outliers: {outliers.values}")
print(f"size : {len(outliers)}")
print(f"lower: {lower_bound}")
print(f"upper: {upper_bound}")

# Confidence interval for mean
confidence = 0.95
n = len(stress_level)
t_value = stats.t.ppf((1 + confidence) / 2, df=n - 1)
lower = mean - t_value * standart_error
upper = mean + t_value * standart_error
print(f"%95 Confidence Interval for Mean: ({lower:.2f}, {upper:.2f})")

# Confidence interval for variance
variance = standart_deviation ** 2
chi2_lower = stats.chi2.ppf((1 - confidence) / 2, df=n - 1)
chi2_upper = stats.chi2.ppf((1 + confidence) / 2, df=n - 1)
variance_lower = (n - 1) * variance / chi2_upper
variance_upper = (n - 1) * variance / chi2_lower
print(f"95% Confidence Interval for Variance: ({variance_lower:.2f}, {variance_upper:.2f})")

# Hypothesis test
hipotez = 6
t_statistic = (mean - hipotez) / standart_error
p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df=n - 1))

print("t-statistic:", t_statistic)
print("p-value:", p_value)

if p_value < 0.05:
    print("Reject H₀: Mean is significantly different from 6.")
else:
    print("Fail to reject H₀: Mean is not significantly different from 6.")


# Sample size calculation
desired_margin = 0.1
z_value = stats.norm.ppf(0.95)
required_n = ((z_value * standart_deviation) / desired_margin) ** 2
print("Required Sample Size (90% CI, margin 0.1):", math.ceil(required_n))
