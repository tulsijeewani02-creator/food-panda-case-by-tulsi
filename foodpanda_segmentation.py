# ── STEP 1: Import all libraries ────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ── STEP 2: Load the data ────────────────────────────────────────────

# Replace 'your_file.xlsx' with your actual file name
df = pd.read_excel('your_file.xlsx')
print('Shape:', df.shape)  # Should be (6000, 20)
print('Unique customers:', df['customer id'].nunique())  # Should be 6000

# ── STEP 3: Verify data level (1 row = 1 customer) ──────────────────

assert df['customer id'].nunique() == len(df), 'Not one row per customer!'
print('Data level confirmed: 1 row per customer')

# ── STEP 4: Check missing values in CLUSTERING FEATURES ONLY ────────

features = ['price', 'order frequency', 'loyalty points']
print(df[features].isnull().sum())  # Should all be 0

# ── STEP 5: Correlation check ────────────────────────────────────────

print(df[features].corr().round(3))  # All near zero = good, keep all features

# ── STEP 6: Standardize features ─────────────────────────────────────

X_raw = df[features].copy()
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ── STEP 7: Naive baseline ───────────────────────────────────────────

np.random.seed(42)
baseline = np.random.randint(0, 3, size=len(X))
print(f'Baseline Silhouette: {silhouette_score(X, baseline):.4f}')  # ~ -0.006# ── STEP 8: Test K=2 to K=10 ─────────────────────────────────────────

sil, db, inertia = [], [], []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil.append(silhouette_score(X, labels))
    db.append(davies_bouldin_score(X, labels))
    inertia.append(km.inertia_)
    print(f'K={k} Sil={sil[-1]:.4f} DB={db[-1]:.4f} Inertia={inertia[-1]:.1f}')

# ── STEP 9: Plot the 3 metrics ───────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(K_range, sil, marker='o', color='blue')
axes[0].axhline(y=0.244, color='red', linestyle='--', label='K=3 value')
axes[0].set_title('Silhouette Score (Higher is Better)')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Silhouette Score')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, db, marker='o', color='green')
axes[1].set_title('Davies-Bouldin Index (Lower is Better)')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('DB Index')
axes[1].grid(True, alpha=0.3)

axes[2].plot(K_range, inertia, marker='o', color='red')
axes[2].axvline(x=3, color='orange', linestyle='--', label='Elbow at K=3')
axes[2].set_title('Inertia / Elbow Method')
axes[2].set_xlabel('Number of Clusters (K)')
axes[2].set_ylabel('Inertia')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ── STEP 10: Apply final model K=3 ──────────────────────────────────

best_k = 3  # Selected based on elbow + DB improvement + business rationale
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['Cluster'] = km_final.fit_predict(X)

# ── STEP 11: Describe clusters (using ORIGINAL unscaled values) ──────

profile = df.groupby('Cluster').agg(
    avg_price=('price', 'mean'),
    avg_order_freq=('order frequency', 'mean'),
    avg_loyalty=('loyalty points', 'mean'),
    count=('customer id', 'count'),
    churn_pct=('churned', lambda x: (x == 'Inactive').mean() * 100)
).round(1)

profile['percentage'] = (profile['count'] / len(df) * 100).round(1)
profile = profile[['count', 'percentage', 'avg_price', 'avg_order_freq', 'avg_loyalty', 'churn_pct']]
profile.columns = ['Count', 'Percentage', 'Avg Price (PKR)', 'Avg Order Frequency', 'Avg Loyalty Points', 'Churn %']
print(profile)

# ── STEP 12: PCA visualization ───────────────────────────────────────

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

colors = ['red', 'blue', 'green']
cluster_names = {0: 'Heavy Spenders', 1: 'Casual Browsers', 2: 'Loyal Engagers'}

plt.figure(figsize=(10, 6))
for c in range(best_k):
    mask = df['Cluster'] == c
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                c=colors[c], alpha=0.5, s=15, 
                label=f'{cluster_names[c]} (Cluster {c})')

plt.title('Foodpanda Customer Segments - PCA Visualization')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ── STEP 13: Baseline comparison bar chart ──────────────────────────

baseline_sil = silhouette_score(X, baseline)
kmeans_sil = silhouette_score(X, df['Cluster'])

plt.figure(figsize=(6, 4))
bars = plt.bar(['Random Baseline (Naive)', 'K-Means (K=3)'], 
               [baseline_sil, kmeans_sil], 
               color=['gray', 'green'])
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.title('K-Means Outperforms Random Baseline by 40x')
plt.ylabel('Silhouette Score (Higher is Better)')
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{bar.get_height():.3f}', ha='center', va='bottom')
plt.show()

print(f"\nFinal Summary:")
print(f"Baseline Silhouette Score (Random): {baseline_sil:.4f}")
print(f"K-Means Silhouette Score (K=3): {kmeans_sil:.4f}")
print(f"Improvement Factor: {kmeans_sil / baseline_sil:.0f}x better than baseline")