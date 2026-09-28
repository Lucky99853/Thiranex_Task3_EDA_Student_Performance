# Thiranex Internship Task 3: Exploratory Data Analysis (EDA) - Student Academic Performance

[![Python Version](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![License: Open Data](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset: UCI ML Repository](https://img.shields.io/badge/Dataset-UCI_ML_Repo-orange.svg)](https://archive.ics.uci.edu/dataset/320/student+performance)
[![Status: Complete & Audited](https://img.shields.io/badge/Audit-PASSED-brightgreen.svg)]()

---

## 1. Project Title & Overview
**Exploratory Data Analysis (EDA) on Secondary School Student Academic Performance**

This repository contains the complete, reproducible end-to-end Exploratory Data Analysis (EDA) project developed for **Thiranex Internship Task 3**. The project investigates the demographic, social, institutional, and behavioral determinants of secondary student academic achievement across two core disciplines: Mathematics and Portuguese Language.

---

## 2. Project Objectives
* **Uncover Patterns & Trajectories:** Investigate student grade distributions across trimesters ($G1, G2, G3$) and profile demographic trends.
* **Rigorous Statistical Testing:** Employ parametric (mean, standard deviation, Welch's unequal variances t-test) and non-parametric (median, IQR, Mann-Whitney U test, Spearman rank correlation, Cohen's $d$ effect sizes) methodologies.
* **Identify Critical Factors:** Discern key factors associated with academic performance, including historical failures, weekly study investment, parental education, digital access, and institutional support.
* **Cross-Cohort Synthesis:** Systematically contrast STEM (Mathematics) against Humanities (Portuguese Language) cohorts.
* **Scientific Rigor & Ethics:** Strictly differentiate statistical correlation from causal impact, adhering to observational research standards.
* **100% Provenance & Reproducibility:** Ensure the raw data remains strictly immutable and every analytical step is automated and reproducible.

---

## 3. Dataset Source & Attribution
* **Repository:** [UCI Machine Learning Repository: Student Performance Data Set](https://archive.ics.uci.edu/dataset/320/student+performance)
* **Citation:** Cortez, P., & Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance*. In A. Brito & J. Teixeira (Eds.), Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008), pp. 5–12, Porto, Portugal. EUROSIS, ISBN 978-9077381-39-7.
* **Data Context:** Survey and administrative records collected during the 2005–2006 academic year from two public secondary schools in the Alentejo region of Portugal: *Gabriel Pereira* (`GP`) and *Mousinho da Silveira* (`MS`).
* **Privacy & Compliance:** Fully anonymized; contains zero Personally Identifiable Information (PII).

---

## 4. Dataset Dimensions & Schema

| Cohort / File | Subject Focus | Raw Records | Processed Records | Discarded Rows | Features | Target Variable |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| [`student-mat-clean.csv`](data/processed/student-mat-clean.csv) | Mathematics | 395 | 395 | **0 (0.0%)** | 33 | `G3` (Final Grade, 0–20) |
| [`student-por-clean.csv`](data/processed/student-por-clean.csv) | Portuguese Language | 649 | 649 | **0 (0.0%)** | 33 | `G3` (Final Grade, 0–20) |

### Attribute Taxonomy (33 Variables)
1. **Academic Performance Outcomes (Continuous/Metric, 0–20 scale):** `G1` (Trimester 1), `G2` (Trimester 2), `G3` (Final Grade / Target).
2. **Attendance & History (Counts):** `absences` (Days absent, 0–75), `failures` (Past class failures, 0–3).
3. **Student Demographics:** `school` (`GP`, `MS`), `sex` (`F`, `M`), `age` (15–22), `address` (`U` - Urban, `R` - Rural).
4. **Family & Social Background:** `famsize` (`LE3`, `GT3`), `Pstatus` (`T` - Together, `A` - Apart), `Medu` (Mother's education, 0–4), `Fedu` (Father's education, 0–4), `Mjob`, `Fjob`, `reason`, `guardian`.
5. **Study Habits & Institutional Support:** `traveltime` (1–4), `studytime` (1: <2h, 2: 2–5h, 3: 5–10h, 4: >10h), `schoolsup` (`yes`/`no`), `famsup` (`yes`/`no`), `paid` (`yes`/`no`), `activities` (`yes`/`no`), `nursery` (`yes`/`no`), `higher` (`yes`/`no`), `internet` (`yes`/`no`), `romantic` (`yes`/`no`).
6. **Health & Lifestyle (1–5 Likert Scales):** `famrel` (Family relationship quality), `freetime` (Free time after school), `goout` (Going out with friends), `Dalc` (Workday alcohol consumption), `Walc` (Weekend alcohol consumption), `health` (Current health status).

---

## 5. Raw-Data Preservation Policy
* **Read-Only Quarantine:** Original source files in [`data/raw/`](data/raw/) (`student-mat.csv`, `student-por.csv`, `student.txt`, `student-merge.R`) are quarantined and treated as strictly immutable.
* **Byte-for-Byte Invariance:** Automated checksum and timestamp validation ensures that no raw data file is ever overwritten, cleaned, modified, or appended to.
* **Zero Arbitrary Deletions:** No records were removed due to outlier status or extreme values.

---

## 6. Data Cleaning & Quality Assessment Methodology
The cleaning process ([`src/data_cleaning.py`](src/data_cleaning.py)) applied defensible quality-assurance protocols:
1. **Completeness & Duplicate Audit:** Confirmed exactly 0 null/missing cells and 0 duplicate rows across both datasets (100% completeness).
2. **Domain & Range Verification:** Validated 100% of values against the official UCI data dictionary; verified that all categorical inputs match documented levels and numerical values fall within physical bounds (e.g. absences $\ge 0$, grades $0 \le G \le 20$).
3. **Investigation of Structural Zeros ($G3 = 0$):**
   - **Math:** 38 students ($9.62\%$); **Portuguese:** 15 students ($2.31\%$).
   - *Discovery:* 100% of students with $G3=0$ had positive prior trimester grades (Math $G1$ mean 7.5, max 12) and **exactly 0 recorded absences**.
   - *Decision:* **Retained.** These records represent administrative exam absences, mid-year unenrollments, or structural dropouts. Dropping them would introduce severe survivorship bias.
4. **Absence Outliers Retention:**
   - Identified 15 outliers in Math (up to 75 days) and 21 in Portuguese (up to 32 days) using the IQR rule ($Q3 + 1.5 \times \text{IQR}$).
   - *Decision:* **Retained.** High absences reflect real-world chronic medical conditions or outside obligations; several high-absence students maintained strong passing marks.
5. **Non-Traditional Age Consistency ($Age \ge 20$):**
   - Verified that older students (5 in Math, 9 in Portuguese) possess multiple historical failures (consistent with Portuguese grade-repetition regulations) or represent high-achieving adult continuing education students.
6. **Processed Dataset Generation:** Cleaned, whitespace-trimmed working copies were exported to [`data/processed/`](data/processed/).

---

## 7. Major EDA Findings

### A. Dominant Influencing Factors Associated with Final Grade ($G3$)
1. **Academic Momentum ($G2$ and $G1$):**
   - Second trimester grade ($G2$) is the strongest overall correlate with final grade ($r = +0.905$ in Math, $r = +0.919$ in Portuguese).
   - First trimester grade ($G1$) is similarly potent ($r = +0.801$ in Math, $r = +0.826$ in Portuguese), providing an early intervention window.
2. **The Higher Education Aspiration Advantage (`higher`):**
   - Aspiration to pursue tertiary education is the single most distinct categorical factor:
     - **Mathematics:** Students aiming for higher ed averaged **10.61** vs. **6.80** for peers without aspirations ($\Delta = +3.81$ points, Cohen's $d = 0.84, p = 0.0024$).
     - **Portuguese:** Students aiming for higher ed averaged **12.28** vs. **8.80** ($\Delta = +3.48$ points, Cohen's $d = 1.14, p = 2.32 \times 10^{-14}$).
3. **Cumulative Burden of Past Failures (`failures`):**
   - Past failure history is the strongest non-grade negative factor ($r = -0.360$ in Math, $r = -0.393$ in Portuguese). Each prior failure depresses final grades by an average of 3 to 5 points.
4. **Study Time Diminishing Returns (`studytime`):**
   - Increasing weekly study time from $<2$ hours to $5$–$10$ hours improves performance in both subjects ($10.05 \rightarrow 11.40$ in Math; $10.84 \rightarrow 13.23$ in Portuguese). Exceeding 10 hours/week shows plateauing or slight regression, indicating struggling students compensating with long study hours.
5. **Parental & Digital Capital (`Medu`, `address`, `internet`):**
   - Mother's education (`Medu`) exhibits persistent positive correlation ($r = 0.22$–$0.24$), outperforming father's education (`Fedu`).
   - Urban students outscore rural peers by $+1.16$ points in Math ($p = 0.0366$) and $+1.18$ points in Portuguese ($p < 0.0001$).
   - Home internet access confers an average $+1.15$ to $+1.21$ point advantage ($p < 0.05$).

### B. Mathematics vs. Portuguese Cohort Divergence
* **Achievement & Dispersion:** Portuguese Language students demonstrate higher overall achievement (Mean $11.91 \pm 3.23$, $84.6\%$ pass rate) compared to Mathematics (Mean $10.42 \pm 4.58$, $67.1\%$ pass rate).
* **Severe Failure Disparity:** Mathematics exhibits a **4.2x higher rate** of severe zero-grades ($9.62\%$ vs. $2.31\%$).
* **Subject-Specific Gender Crossover:**
  - **Mathematics:** Male students outscore female students (Mean $10.91$ vs. $9.97$, $\Delta = +0.95, p = 0.0396$).
  - **Portuguese:** Female students outscore male students (Mean $12.25$ vs. $11.41$, $\Delta = +0.85, p = 0.0011$).
* **Institutional Variation:** Gabriel Pereira (`GP`) substantially outperforms Mousinho da Silveira (`MS`) in Portuguese Language ($\Delta = +1.93, p < 10^{-10}$), while the difference in Math is not statistically significant ($\Delta = +0.64, p = 0.343$).

---

## 8. Essential Methodological Limitation: Correlation vs. Causation
> [!IMPORTANT]
> **Correlation Does Not Imply Causation.**  
> All findings in this project are based on cross-sectional observational survey and administrative data. No controlled randomized experiments were conducted.
>
> A prime demonstration is **School Educational Support (`schoolsup`)**:
> - Students receiving extra school support achieve **significantly lower** average grades in both Math ($9.43$ vs. $10.56, p = 0.0197$) and Portuguese ($11.28$ vs. $11.98, p = 0.0268$).
> - **Scientific Explanation (Confounding by Indication):** Remedial tutoring does not cause academic impairment. Rather, students experiencing prior academic distress are preferentially selected into remedial programs.
>
> Similarly, observed differences across parental education, internet access, and urban geography reflect systemic socioeconomic compounding rather than isolated causal mechanisms.

---

## 9. Project Structure

```text
Thiranex_Task3_EDA_Student_Performance/
├── data/
│   ├── raw/                                     # Immutable source data (quarantined)
│   │   ├── student-mat.csv                      # Mathematics raw records (N=395)
│   │   ├── student-por.csv                      # Portuguese raw records (N=649)
│   │   ├── student.txt                          # Official attribute documentation
│   │   └── student-merge.R                      # Demographic matching script
│   └── processed/                               # Cleaned, standardized working datasets
│       ├── student-mat-clean.csv                # Validated Math data (N=395, 33 cols)
│       └── student-por-clean.csv                # Validated Portuguese data (N=649, 33 cols)
├── notebooks/                                   # Exploratory Jupyter notebooks
├── src/                                         # Automated, reproducible pipelines
│   ├── data_cleaning.py                         # Cleaning, domain validation & quality audit
│   ├── eda_analysis.py                          # Full statistical profiling, tests & plots
│   └── eda_baseline.py                          # Initial baseline profiling script
├── outputs/                                     # Machine-readable tables & text summaries
│   ├── baseline_data_quality.csv                # Baseline null & duplicate audit
│   ├── column_schema_audit.csv                  # 33-feature schema & dtype audit
│   ├── numerical_descriptive_stats.csv          # Extended numerical descriptive metrics
│   ├── categorical_frequency_stats.csv          # Categorical mode & frequency tables
│   ├── key_variables_inspection.csv             # Comparative focal feature profiles
│   ├── student_overlap_analysis.csv             # Co-enrolled student matching (N=382)
│   ├── data_quality_after_cleaning.csv          # Post-cleaning verification audit
│   ├── descriptive_statistics_math.csv          # Final descriptive stats for Math
│   ├── descriptive_statistics_portuguese.csv    # Final descriptive stats for Portuguese
│   ├── correlation_matrix_math.csv              # Pearson correlation matrix (Math)
│   ├── correlation_matrix_portuguese.csv        # Pearson correlation matrix (Portuguese)
│   ├── group_comparisons.csv                    # 24 Welch t-test & Cohen's d evaluations
│   ├── data_cleaning_report.txt                 # Detailed cleaning audit narrative
│   ├── eda_report.txt                           # Comprehensive narrative EDA report
│   ├── eda_quality_control_report.txt           # Independent QA & verification audit
│   └── figures/                                 # 16 High-resolution 300 DPI figures
│       ├── 01_baseline_grade_distributions.png
│       ├── 02_target_g3_zero_scores.png
│       ├── 03_baseline_demographics_and_study_habits.png
│       ├── 04_absences_and_studytime_profiles.png
│       ├── 05_data_quality_audit_summary.png
│       ├── 06_g3_zero_investigation_deepdive.png
│       ├── 07_absence_outliers_and_age_consistency.png
│       ├── 08_univariate_grade_distributions.png
│       ├── 09_univariate_demographics_and_attendance.png
│       ├── 10_bivariate_academic_progression_g1_g2_vs_g3.png
│       ├── 11_bivariate_study_habits_and_failures_vs_g3.png
│       ├── 12_bivariate_absences_vs_g3.png
│       ├── 13_correlation_heatmaps_math_vs_portuguese.png
│       ├── 14_categorical_factors_vs_g3.png
│       ├── 15_demographic_and_school_group_comparisons.png
│       └── 16_math_vs_portuguese_comprehensive_comparison.png
├── reports/                                     # Structured internship reporting
└── README.md                                    # Comprehensive project documentation
```

---

## 10. Reproducibility & Execution Instructions

### Prerequisites
* Python 3.10+ (Tested on Python 3.14)
* Core libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`

Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scipy
```

### Reproducing the Complete Pipeline
From the project root directory:

1. **Execute Data Cleaning & Quality Audit:**
   ```bash
   python src/data_cleaning.py
   ```
   *Reads from `data/raw/`, validates schema and domains, verifies $G3=0$ cases and outliers without deleting rows, and generates `data/processed/`.*

2. **Execute Full Exploratory Data Analysis:**
   ```bash
   python src/eda_analysis.py
   ```
   *Generates descriptive statistics, correlation matrices, Welch t-tests, Cohen's $d$ values, narrative reports, and all 9 analytical figures (08 to 16) in `outputs/figures/`.*

3. **Verify Quality Control Audit:**
   Inspect [`outputs/eda_quality_control_report.txt`](outputs/eda_quality_control_report.txt) to verify that all numerical calculations match the underlying data.
