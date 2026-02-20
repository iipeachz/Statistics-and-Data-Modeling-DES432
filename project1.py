import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ตั้งค่าสไตล์
sns.set_theme(style="whitegrid")

# =====================
# 1. LOAD & CLEAN DATA
# =====================
file_path = "Provisional_COVID-19_death_counts_and_rates_by_month,_jurisdiction_of_residence,_and_demographic_characteristics_20260220.csv"
df = pd.read_csv(file_path)

# [Inconsistent Entries] - ลบ comma ออกจากตัวเลขที่เป็น string
df['year_clean'] = df['year'].astype(str).str.replace(',', '').str.strip()
df['deaths_clean'] = df['COVID_deaths'].astype(str).str.replace(',', '').str.strip()

# [Missing Values] - แปลงเป็น numeric และจัดการค่าว่าง
df['year_clean'] = pd.to_numeric(df['year_clean'], errors='coerce')
df['deaths_clean'] = pd.to_numeric(df['deaths_clean'], errors='coerce')

# กรองเฉพาะ United States และกลุ่มประชากรที่เหมาะสม (Sex) เพื่อป้องกันการนับซ้ำ
df_us = df[(df['jurisdiction_residence'] == "United States") & (df['group'] == 'Sex')].copy()
df_us = df_us.dropna(subset=['deaths_clean', 'year_clean'])

# =====================
# 2. DESCRIPTIVE STATISTICS
# =====================
mean_val = df_us['deaths_clean'].mean()
median_val = df_us['deaths_clean'].median()
std_val = df_us['deaths_clean'].std()
q1 = df_us['deaths_clean'].quantile(0.25)
q3 = df_us['deaths_clean'].quantile(0.75)
iqr_val = q3 - q1

print("\n========== 4. DESCRIPTIVE STATISTICS ==========")
print(f"Measures of Center (ค่ากลาง):")
print(f" - Mean (ค่าเฉลี่ย): {mean_val:,.2f}")
print(f" - Median (ค่ามัธยฐาน): {median_val:,.2f}")
print(f"\nMeasures of Spread (การกระจายตัว):")
print(f" - Standard Deviation (ส่วนเบี่ยงเบนมาตรฐาน): {std_val:,.2f}")
print(f" - IQR (ช่วงระหว่างควอไทล์): {iqr_val:,.2f}")
print(f" - Range: {df_us['deaths_clean'].min()} to {df_us['deaths_clean'].max()}")
print("===============================================")
print("Interpretation: ค่าเฉลี่ยสูงกว่าค่ามัธยฐานอย่างมาก สะท้อนว่าข้อมูลเป็นบวกเบ้ (Right-skewed) "
      "และค่า Standard Deviation ที่สูงแสดงให้เห็นว่ายอดผู้เสียชีวิตในแต่ละเดือนมีความแตกต่างกันมากตามระลอกการระบาด")

# =====================
# 3. EDA: VISUAL ANALYSIS
# =====================

# --- 3.1 Distribution Analysis (Histogram & Boxplot) ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df_us['deaths_clean'], kde=True, color='teal')
plt.title('Distribution of Monthly Deaths')
plt.subplot(1, 2, 2)
sns.boxplot(x=df_us['deaths_clean'], color='coral')
plt.title('Boxplot of Monthly Deaths')
plt.tight_layout()
plt.show()
# Interpretation: Histogram แสดงให้เห็นความถี่ที่กระจุกตัวอยู่ในช่วงค่าน้อย 
# แต่ Boxplot แสดงให้เห็น Outliers (จุดไข่ปลา) จำนวนมากในช่วงค่าสูง ซึ่งคือช่วงวิกฤตการระบาด

# --- 3.2 Group Comparisons (Side-by-side Boxplots) ---
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_us, x='subgroup1', y='deaths_clean', palette='Set2')
plt.title('Comparison: Male vs Female COVID Deaths')
plt.ylabel('Monthly Deaths')
plt.show()
# Interpretation: เมื่อเปรียบเทียบระหว่างเพศชายและหญิง พบว่าการกระจายตัวของข้อมูลค่อนข้างใกล้เคียงกัน 
# แต่เพศชายมักจะมีค่าเฉลี่ยและค่าสูงสุด (Peak) ที่สูงกว่าเพศหญิงเล็กน้อยในหลายช่วงเวลา

# --- 3.3 Relationship Exploration (Scatter Plot) ---
# ดูความสัมพันธ์กับเวลา (Time Progress)
df_trend = df_us.groupby(['year_clean', 'month'])['deaths_clean'].sum().reset_index()
df_trend['month_index'] = np.arange(len(df_trend))
plt.figure(figsize=(10, 6))
sns.regplot(data=df_trend, x='month_index', y='deaths_clean', color='purple', scatter_kws={'alpha':0.6})
plt.title('Relationship: Time Progress vs Total Monthly Deaths')
plt.xlabel('Month Index (from Jan 2020)')
plt.ylabel('Total Deaths (Summed)')
plt.show()
# Interpretation: Scatterplot และ Regression line แสดงให้เห็นว่าไม่มีความสัมพันธ์เชิงเส้นที่ชัดเจน 
# ข้อมูลมีความผันผวนสูงตามคลื่นการระบาด (Waves) มากกว่าที่จะเพิ่มขึ้นหรือลดลงเป็นเส้นตรงตามเวลา