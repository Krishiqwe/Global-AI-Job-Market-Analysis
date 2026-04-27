
# IMPORTS
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
# ─────────────────────────────────────────────────────────────────
# SECTION 1 – INTRODUCTION
# ─────────────────────────────────────────────────────────────────

print("  SECTION 1 – INTRODUCTION")
print("=" * 72)
print("""
The rapid integration of Artificial Intelligence into every industry
has created a dynamic and fast-growing job market. This project aims
to analyse the 'Global AI Job Market & Salary Trends 2025' dataset
to extract meaningful insights about job roles, required skills, salary
distributions, and hiring patterns across countries and experience
levels.

OBJECTIVES:
  1. Perform Exploratory Data Analysis (EDA) to understand the dataset's
     structure, distributions, and key trends.
  2. Conduct Hypothesis Testing to determine whether remote-work
     availability significantly affects salaries.
  3. Apply Simple & Multiple Linear Regression to predict salary based
     on experience level and other features.
  4. Build a Logistic Regression classifier to predict whether a job
     offers remote work.
  5. Perform K-Means Clustering to segment job postings into
     meaningful groups by salary and demand.
  6. Analyse correlation among numerical features and draw actionable
     business insights.
""")

# ─────────────────────────────────────────────────────────────────
# SECTION 2 – SOURCE OF DATASET
# ─────────────────────────────────────────────────────────────────
data = pd.read_csv("ai_job_dataset.csv")
data.sample(5)
#EDA:
data.info()
data.describe()
data.isnull().sum()
print(data.duplicated().sum())
numeric_columns = data[['salary_usd','remote_ratio','years_experience','job_description_length','benefits_score']]
corr_matrix = numeric_columns.corr()
#heatMap:
plt.figure(figsize=(8, 6))
sb.heatmap(corr_matrix, annot=True)

plt.title('Correlation Matrix of Job Factors')
plt.show()
#Univariate Analysis:
data.sample(5)
sb.countplot(data['job_title'])
data['job_title'].value_counts().max()
#Ml engineers are maximum in count
#Salary in USD:
data['salary_usd'].max()
sb.scatterplot(data['salary_usd'])
sb.histplot(data = data , x ='salary_usd',bins = 15,kde=True)
# sb.histplot(data=data,x='years_experience',bins = 10,kde=True)
#Experience level:
experience = data['experience_level'].value_counts()
experience
plt.pie(experience.values,labels= experience.index,autopct='%1.1f%%')
plt.show()
plt.pie(experience.values,labels= experience.index,autopct='%1.1f%%')
plt.show()
#Employment type:
plt.pie(data['employment_type'].value_counts().values,labels= data['employment_type'].value_counts().index,autopct='%1.1f%%')
plt.show()
#Company Location:
sb.countplot(data['company_location'])
company_size = data['company_size'].value_counts()
print(company_size)
plt.pie(data['company_size'].value_counts().values,labels = data['company_size'].value_counts().index,autopct='%1.1f%%')
plt.show()
#Industry:
industry = data['industry'].value_counts()
# print(industry)
# print(industry.values.min())
data['industry'].value_counts().plot(kind='bar')
#OutLiers:
sb.boxplot(data['salary_usd'])
perecentile25 = data['salary_usd'].quantile(0.25)
percentile75 = data['salary_usd'].quantile(0.75)
print(perecentile25,percentile75)
iqr = percentile75 - perecentile25
print(iqr)
lower_limit = perecentile25 - 1.5*iqr
upper_limit = percentile75 + 1.5*iqr
print(lower_limit,upper_limit)
#Finding outliers:
data[data['salary_usd'] > upper_limit].shape[0]
data[data['salary_usd'] < lower_limit].shape[0]
new_df = data[data['salary_usd'] < upper_limit]
new_df.sample(7)
sb.histplot(data=new_df,x='salary_usd',bins = 10,kde=True)
sb.boxplot(new_df['salary_usd'])
new_df.shape
sb.boxplot(data['years_experience'])
#Bivariate Analysis:
median_salary = data.groupby('job_title')['salary_usd'].median().sort_values().plot(kind='bar')
plt.xlabel('Job title')
plt.ylabel('median salary in USD')
plt.title('Median salary by job Title')
plt.show()
data.groupby('experience_level')['salary_usd'].mean().sort_values().plot(kind='bar')
plt.ylabel('Average salary in USD')
plt.xlabel('Experience Level')
plt.title('Average salary by Experince level')
plt.show()
#Multivariate Analysis:
experience_level=["EN","MI","SE","EX"]
sb.barplot(data=data ,x="experience_level" ,y="salary_usd" ,hue="education_required" ,palette="viridis")
plt.title("Average Salary by experience_level and education_required" ,fontsize=15)
plt.xlabel("experience_level" ,fontsize=12)
plt.ylabel("salary_usd" ,fontsize=12)
plt.legend(title="education_required")
plt.show()
plt.figure(figsize=(10,6))
sb.scatterplot(data =data ,x="employment_type" ,y="salary_usd" ,color="darkred")
plt.title("Relationship Between employment_type and salary_usd")
plt.xlabel("employment_type")
plt.ylabel("salary_usd")
plt.show()
plt.figure(figsize=(10, 6))
sb.scatterplot(x='years_experience', y='salary_usd', data=data,color='darkred')
plt.title('Relationship Between Required Years of Experience and Salary')
plt.xlabel('Required Years of Experience')
plt.ylabel('Annual Salary (USD)')
plt.show()
plt.figure(figsize=(12,6))
sb.lineplot(x="years_experience",y="salary_usd",color="blue",data=data,marker="o")
plt.title("salary by years of exepriance")
plt.xlabel("Years of Experience")
plt.ylabel("Salary (USD)")
plt.show()
skills = data['required_skills'].str.split(', ').explode()
top_skills = skills.value_counts().head(10)
plt.figure(figsize=(10,5))
sb.barplot(x=top_skills.values, y=top_skills.index)
plt.title("Top 10 AI Skills")
plt.xlabel("Count")
plt.ylabel("Skill")
plt.show()
education_required =["Bachelor","Master","Associate","PhD"]
plt.figure(figsize=(10,6))
sb.boxplot(data=data,x="education_required" ,y="salary_usd")
plt.title("Salary Distribution (USD) by education_required")
plt.xlabel("education_required")
plt.ylabel("salary_usd")
plt.show()
# ANALYSIS 1 – HYPOTHESIS TESTING
# Does remote work significantly affect salary?
#Jobs that are fully remote
remote_group = data[data["remote_ratio"] == 100]["salary_usd"]
#Jobs that are fully on-site
onsite_group = data[data["remote_ratio"] == 0]["salary_usd"]
#Hybrid jobs
hybrid_group = data[data["remote_ratio"] == 50]["salary_usd"]
hybrid_group
remote_count = len(remote_group)
onsite_count = len(onsite_group)
remote_mean = remote_group.mean()
onsite_mean = onsite_group.mean()
 
remote_std = remote_group.std()
onsite_std = onsite_group.std()
onsite_std
from scipy import stats
t_stat, p_value = stats.ttest_ind(remote_group, onsite_group, equal_var=False)
print("ANALYSIS RESULTS:")
print(f"Remote jobs Count:{remote_count}")
print(f"Remote jobs Mean: ${remote_mean:,.0f}")
print(f"Remote jobs Std: ${remote_std:,.0f}")
print(f"On-site jobs Count: {onsite_count}")
print(f"On-site jobs Mean: ${onsite_mean:,.0f}")
print(f"On-site jobs Std: ${onsite_std:,.0f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
# Decision based on p-value:
if p_value < 0.05:
    print("DECISION : Reject null hyposthesis")
    print("MEANING  : Remote work DOES significantly affect salary.")
else:
    print("DECISION : Fail to Reject Hypothesis")
    print("MEANING  : No significant salary difference between remote and on-site.")
#plt.figure(figsize=(10, 6))
sb.boxplot(x='remote_ratio', y='salary_usd', data=data)
plt.title('Salary Distribution by Remote Work Ratio', fontsize=15)
plt.xlabel('Remote Ratio (%)', fontsize=12)
plt.ylabel('Salary (USD)', fontsize=12)
plt.show()
#Objective 2: Linear Regression Salary prediction:
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
df_lr = new_df.copy()
exp_mapping = {"EN": 0, "MI": 1, "SE": 2, "EX": 3}
df_lr["experience_num"] = df_lr["experience_level"].map(exp_mapping)
# company_size:
size_mapping = {"S": 0, "M": 1, "L": 2}
df_lr["size_num"] = df_lr["company_size"].map(size_mapping)

print("Experience level mapping used:")
print(exp_mapping)
print("Company size mapping used:")
print(size_mapping)
df_lr.head(5)
#Input feature:
x = df_lr[['experience_num','size_num','years_experience','remote_ratio']]
y = df_lr['salary_usd']
print("Input features: ",x.shape)
print("Output features: ",y.shape)
#Train-Test:
X_train, X_test, y_train, y_test = train_test_split(x, y,test_size = 0.2,random_state = 42)
print("Training set size :", len(X_train),"rows")
print("Testing  set size :", len(X_test),"rows")
#Training the model:
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training done!")
print()
#Prediction:
y_predicted = model.predict(X_test)
r2   = r2_score(y_test, y_predicted)
mae  = mean_absolute_error(y_test, y_predicted)
rmse = np.sqrt(mean_squared_error(y_test, y_predicted))

print("ANALYSIS RESULTS:")
print(f"R2 = {r2:.4f} : Model explains {r2*100:.1f}% of salary variation")
print(f"MAE = ${mae:.0f} : Average prediction error")
print(f"RMSE = ${rmse:.0f} : Root mean square error")
plt.figure(figsize=(10, 6))
sb.scatterplot(x=y_test, y=y_predicted, alpha=0.6, color='blue')

# Add a diagonal line for reference (Perfect Prediction)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         color='red', linestyle='--', lw=2, label='Perfect Prediction')
plt.title('How Accurate is our Model? (Actual vs. Predicted)', fontsize=15)
plt.xlabel('Actual Salary (USD)')
plt.ylabel('Predicted Salary (USD)')
plt.show()
#Objective 3:
#Correlation:
# Is there a relationship between years of experience and salary:
print("OBJECTIVE 3 — Correlation: Years of Experience vs Salary")
corr_coeff, p_value2 = stats.pearsonr(data["years_experience"], data["salary_usd"])
print("Correlation Coefficient :",round(corr_coeff, 4))
print("P-value :",round(p_value2, 6))

# Interpret the correlation strength
if abs(corr_coeff) >= 0.7:
    strength = "Strong"
elif abs(corr_coeff) >= 0.4:
    strength = "Moderate"
else:
    strength = "Weak"
direction = "Positive" if corr_coeff > 0 else "Negative"
print(f"Interpretation: {strength} {direction} Correlation")

if p_value2 < 0.05:
    print("CONCLUSION: p-value < 0.05 : REJECT H0")
    print("Years of experience and salary ARE significantly correlated.")
else:
    print("CONCLUSION: p-value >= 0.05 : FAIL TO REJECT H0")
    print("No significant correlation found.")
#Scatter plot:
avg_by_year = data.groupby("years_experience")["salary_usd"].mean()
plt.figure(figsize=(9, 5))
plt.scatter(data["years_experience"], data["salary_usd"],alpha=0.2, color="steelblue", s=10, label="Individual Jobs")
plt.plot(avg_by_year.index, avg_by_year.values,color="red", linewidth=2, marker="o", label="Average Salary")
plt.title("Years of Experience vs Salary (r = " + str(round(corr_coeff, 3)) + ")",)
plt.xlabel("Years of Experience")
plt.ylabel("Salary (USD)")
plt.show()
#Objective 4:
# Comparing Benefits Score between Small and Large Companies
small_companies = data[data['company_size'] == 'S']
large_companies = data[data['company_size'] == 'L']

scores_small = small_companies['benefits_score']
scores_large = large_companies['benefits_score']

# Run the T-test
t_stat, p_value = stats.ttest_ind(scores_small, scores_large)
print(f"T-Test P-value: {p_value:.4f}")
if p_value < 0.05:
    print("Conclusion : REJECT the null hypothesis.")
    print("Meaning: There is a statistically significant difference.")
else:
    print("Conclusion: FAIL TO REJECT the null hypothesis.")
    print("Meaning: There is NO statistically significant difference.")
# Visualization:
plt.figure(figsize=(8, 6))
plot_data = data[data['company_size'].isin(['S', 'L'])] # Get only S and L rows for the plot
sb.boxplot(x='company_size', y='benefits_score', data=plot_data, order=['S', 'L'],)
plt.title('Benefits Score: Small vs Large Companies')
plt.xlabel('Company Size (S = Small, L = Large)')
plt.ylabel('Benefits Score')
plt.show()
#Objective 5:
# 2. Extract Features (X) and Target (y)
X = data[['years_experience']] 
y = data['job_description_length']
# 3. Split the data into Training and Testing sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training done!")
# 5. Testing and Evaluation
y_pred = model.predict(X_test)

# Extract the mathematical details the model learned
slope = model.coef_[0]
intercept = model.intercept_
r2 = r2_score(y_test, y_pred)

print("Objective 3: Machine Learning Model Results")
print("Predicting: Job Description Length based on Years of Experience")
print(f"Slope (m) : {slope:.4f}")
print(f"Intercept (b): {intercept:.4f}")
print(f"Test R-squared: {r2:.5f}")

# 6. VISUALIZATION
plt.figure(figsize=(8, 6))
sb.scatterplot(x=X_test['years_experience'], y=y_test,alpha=0.5, label='Actual Test Data')
plt.plot(X_test['years_experience'], y_pred, color='black', linewidth=2, label='ML Regression Line')
plt.title('ML Linear Regression: Experience vs. Length')
plt.xlabel('Required Years of Experience')
plt.ylabel('Job Description Length')
plt.show()
