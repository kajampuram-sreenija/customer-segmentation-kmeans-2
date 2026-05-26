# customer-segmentation-kmeans-2
A Python-based data science project that generates synthetic customer data, uses K-Means clustering to segment buyers, and extracts actionable marketing personas.
import pandas as pd
import numpy as np

np.random.seed(42)


data = {
    'CustomerID': range(1, 101),
    'Age': np.random.randint(18, 70, size=100),                  # Ages between 18 and 70
    'Annual_Income_k': np.random.randint(15, 120, size=100),     # Income between $15k and $120k
    'Spending_Score': np.random.randint(1, 100, size=100)        # Score between 1 and 100
}

# 2. Turn this data into a structured table (a DataFrame)
df = pd.DataFrame(data)

# 3. Save it as a spreadsheet file named 'customers.csv'
df.to_csv('customers.csv', index=False)

# 4. Show us the first 5 rows to confirm it worked
print("--- DATA GENERATED SUCCESSFULLY ---")
print(df.head())
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Load the generated dataset
try:
    df = pd.read_csv('customers.csv')
    print("--- DATA LOADED SUCCESSFULLY ---")
except FileNotFoundError:
    print("Error: 'customers.csv' not found. Run your data generator script first!")
    exit()

# 2. Select and Scale Features
# We use Age, Income, and Spending Score. K-Means needs scaled data to perform well.
features = ['Age', 'Annual_Income_k', 'Spending_Score']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Find Optimal Clusters using the Elbow Method
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Curve to help you choose 'K' visually
plt.figure(figsize=(8, 4))
plt.plot(K_range, inertia, marker='o', linestyle='--', color='b')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (Within-cluster variance)')
plt.title('The Elbow Method for Optimal K')
plt.tight_layout()
plt.savefig('elbow_plot.png')  # Saves the plot to your folder
print("- Elbow plot saved as 'elbow_plot.png'")

# 4. Train Final K-Means Model
# Based on typical customer behavior datasets, 4 or 5 is standard. Let's use 4 for this random distribution.
optimal_k = 4 
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)
# 5. Analyze the Profiles (Calculate Averages)
print("\n--- CUSTOMER SEGMENT PROFILES (AVERAGES) ---")

pd.set_option('display.max_columns', None)  

profiles = df.groupby('Cluster')[['Age', 'Annual_Income_k', 'Spending_Score']].mean()
profiles['Customer_Count'] = df['Cluster'].value_counts()
print(profiles.round(1))

# 6. Visualize the Segments
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df, 
    x='Annual_Income_k', 
    y='Spending_Score', 
    hue='Cluster', 
    palette='Set1', 
    s=100, 
    style='Cluster'
)
plt.title('Customer Segments: Income vs Spending Score')
plt.xlabel('Annual Income ($k)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster Group')
plt.tight_layout()
plt.savefig('customer_segments.png')  # Saves your final segment visualization
print("- Segment scatter plot saved as 'customer_segments.png'")
print("\n--- PROCESS COMPLETE ---")
