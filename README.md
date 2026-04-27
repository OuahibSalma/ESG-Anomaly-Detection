# 🌍 ESG Anomaly Detection — A Beginner's Journey

> *Can we trust ESG scores ? This project tries to find out — using machine learning to detect companies that look suspicious.*

---

## 👋 What is this project about ?

ESG stands for **Environmental, Social, and Governance**. Companies receive ESG scores to measure how responsible they are — how much they pollute, how they treat their employees, how transparent their leadership is.

But here is the uncomfortable truth : **ESG scores are mostly self-reported**. Companies fill in their own data. So the question becomes — are some companies lying, exaggerating, or just wildly inconsistent ?

This project uses **unsupervised machine learning** (meaning we have no labels, no "right answers") to automatically detect companies that behave differently from the rest. We call these companies **anomalies**.
---
## 🧭 The Journey — Step by Step
### 1. 🔍 Exploratory Data Analysis (EDA)

Before building any model, we explored the data visually to understand what we are working with.

**What we found :**

- **Carbon Emissions are extremely skewed** — most companies emit very little, but a few emit enormous amounts. This is why we used a **log scale** in our plots. Without it, all the points were crushed against zero and we couldn't see anything useful.

- **ESG scores are surprisingly uniform** — almost every industry hovers around 50–60, regardless of how much they actually pollute. Consumer Goods for example emits far more carbon than any other sector, yet its average ESG score is similar to Technology or Finance.

- **ESG scores have been rising every year from 2015 to 2025** — which sounds great, but given the above, it likely reflects companies getting better at *reporting* their efforts rather than actually becoming more responsible.

> 💡 **Key insight** : ESG scores appear disconnected from actual emissions and financial performance. This is exactly why anomaly detection matters here.

---

### 2. 🛠️ Feature Engineering

Raw data is rarely enough. We created new features to better capture suspicious behavior :

| Feature | What it means |
|---|---|
| `carbon_intensity` | Emissions divided by revenue — how much does a company pollute per dollar earned ? |
| `esg_inconsistency` | Gap between ESG sub-scores — a company that scores 90 on Environment but 10 on Governance is suspicious |
These engineered features are often more informative than the raw ones because they capture **relationships**, not just absolute values.

---

### 3. 🤖 The Three Models

We used three different anomaly detection algorithms. Each one has its own way of deciding what is "normal".

#### 🌲 Isolation Forest
Randomly cuts the data like a tree. Points that get isolated quickly (in fewer cuts) are more likely to be anomalies. Think of it as : *"how easy is it to separate this company from everyone else ?"*

**Key parameter** : `contamination=0.05` — we tell the model to flag the 5% most extreme points. This is a human assumption, not something the model learns.

#### 📍 Local Outlier Factor (LOF)
Compares each company to its neighbors. If a company's neighborhood is very dense but the company itself is far away, it is an outlier. Think of it as : *"does this company fit in with similar companies ?"*

**Why we trust LOF the most** : In our sensitivity check, LOF always detected exactly **550 anomalies** regardless of whether we used 10, 20, or 30 neighbors. That stability is a strong signal.

#### 🔵 DBSCAN
Groups points into dense clusters and labels everything else as noise. It completely failed on this dataset — it put almost all companies into one giant cluster and labeled only 508 as noise, which tells us the data has no clear density-based structure.

**Sensitivity check result for DBSCAN :**
- eps=0.5 → 10,987 noise points
- eps=1.0 → 3,298 noise points
- eps=1.5 → 508 noise points

Way too sensitive to the parameter. Not reliable here.

---

### 4. ⚙️ Hyperparameter Sensitivity Check

Before trusting any model, we tested whether changing the parameters significantly changes the results :

**Isolation Forest :**
- contamination=0.03 → 330 anomalies
- contamination=0.05 → 550 anomalies
- contamination=0.08 → 880 anomalies

Expected — this parameter directly controls how many anomalies are flagged.

**LOF :**
- n_neighbors=10 → 550 anomalies
- n_neighbors=20 → 550 anomalies
- n_neighbors=30 → 550 anomalies

Perfectly stable ✅

**DBSCAN :** Extremely sensitive as shown above ❌

---

### 5. 🤝 Two Approaches to Anomaly Analysis

Since we have no ground truth (no one told us which companies are actually fraudulent), we used two complementary approaches :

#### Approach 1 — LOF Only
We used LOF as our primary detector since it proved the most stable. We then :
- Compared the average profile of anomalies vs normal companies
- Visualized differences with boxplots
- Found the top 20 most extreme anomalies by LOF score
- Checked which industries and years had the most anomalies

#### Approach 2 — Consensus Anomalies (3 models)
A company is flagged as a consensus anomaly only if **all three models agree** it is suspicious.

- With 2 models (Isolation Forest + LOF) : **148 records across 48 companies**
- With 3 models (adding DBSCAN) : much fewer — but these are the **highest confidence** cases

> 💡 In a real business setting, the consensus list is what you bring to a domain expert or stakeholder and say : *"every single model we ran flagged these companies — something is definitely worth investigating here."*

---

### 6. 🗺️ PCA Visualization

With 12 features it is impossible to plot the data directly. We used **PCA (Principal Component Analysis)** to compress everything into 2 dimensions and visualize where anomalies sit.

This is one of the most intuitive ways to validate unsupervised results — if anomalies appear randomly scattered everywhere, the model probably isn't finding real patterns. If they cluster at the edges, that's a good sign.

---

### 7. 🔎 SHAP — Why is a company anomalous ?

Knowing *who* is anomalous is not enough. We used **SHAP values** to explain *why* each company was flagged — which features pushed its anomaly score the highest.

This bridges the gap between *"company X is anomalous"* and *"company X is anomalous because of unusually high carbon intensity and a large gap between its environmental and governance scores."*

---

## 📊 Key Findings

- ESG scores are **not reliable indicators** of actual environmental behavior
- **Consumer Goods** stands out as the highest-emitting sector, yet maintains average ESG scores
- **LOF is the most stable** anomaly detector on this dataset
- **DBSCAN is not suited** for this type of continuous, high-dimensional data
- The rising ESG trend over 2015–2025 likely reflects **better reporting**, not better behavior

---

## ⚠️ Limitations

This project is fully **unsupervised** — there are no labels, no ground truth, no way to say with certainty that a flagged company is actually doing something wrong.

In a real business setting, the next steps would be :
1. Bring the top anomalies to a **domain expert** for review
2. Cross-check against public ESG controversy databases like **RepRisk** or **MSCI ESG Ratings**
3. Request internal data or filings from flagged companies for deeper investigation

Agreement with external sources would increase confidence in the model's output.

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/your-username/esg-anomaly-detection.git
cd esg-anomaly-detection

# Install dependencies
pip install -r requirements.txt

# Open the notebook
jupyter notebook notebook.ipynb
```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
shap
jupyter
```

---

## 🧠 What I Learned

This project taught me that :

- **Unsupervised learning is hard** — without labels, validating your results requires creativity and domain knowledge
- **Stability matters** — a model that gives wildly different results when you change one parameter slightly is not trustworthy
- **Feature engineering is often more important than the model itself** — the `esg_inconsistency` and `carbon_intensity` features captured things raw columns never could
- **Visualization is not optional** — every step of this project became clearer once we plotted it
- **Data can tell a story that contradicts common sense** — we expected polluters to have bad ESG scores. The data said otherwise.

---

## 👤 Author

Made with curiosity and a lot of confusion by a junior data science student learning the ropes. 
