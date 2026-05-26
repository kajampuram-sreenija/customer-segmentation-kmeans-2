Customer Segmentation Project

An unsupervised machine learning project that uses K-Means clustering to partition a customer base into 4 actionable buyer personas based on Age, Annual Income, and Spending Score. 

📊 Key Insights

| Cluster | Persona Nickname | Avg Age | Avg Income | Avg Spend Score | Marketing Strategy |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **0** | **The Affluent Seniors** | 59.9 | \$57.9k | 74.1 | Premium loyalty programs, high-end upgrades. |
| **1** | **Young High-Earners** | 31.1 | \$99.4k | 65.9 | Exclusive VIP launches, trend-driven ads. |
| **2** | **Frugal Youth** | 33.9 | \$34.1k | 41.2 | "Buy Now, Pay Later" (BNPL), flash sales. |
| **3** | **Wealthy Savers** | 44.6 | \$92.3k | 15.1 | Value-focused campaigns, practical utility. |

🛠️ Tech Stack
* Python 3.14+
* Scikit-Learn (K-Means Clustering, StandardScaler)
* Pandas & NumPy
* Matplotlib & Seaborn

 📂 Project Outputs
* `customers.csv`: The generated synthetic customer dataset.
* `elbow_plot.png`: Line chart used to find the optimal cluster count.
* `customer_segments.png`: Final scatter plot visualizing the 4 distinct groups.
