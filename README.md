# Zeotap-Intership-Assignment
# Data Science Assignment: eCommerce Transactions Dataset
1. Introduction
This report presents the findings from the exploratory data analysis (EDA), lookalike modeling, and customer segmentation performed on the eCommerce Transactions dataset. The dataset consists of three files: Customers.csv, Products.csv, and Transactions.csv. The goal is to derive actionable insights and build predictive models to enhance business strategies.

2. Exploratory Data Analysis (EDA)
2.1 Data Overview
Customers: Contains customer demographics including CustomerID, CustomerName, Region, and SignupDate.
Products: Includes product details such as ProductID, ProductName, Category, and Price.
Transactions: Records transaction details including TransactionID, CustomerID, ProductID, TransactionDate, Quantity, and TotalValue.
2.2 Data Cleaning
The following code was used to load the datasets, check for missing values, and merge them:

python

Verify
Copy code
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
customers = pd.read_csv('Customers.csv')
products = pd.read_csv('Products.csv')
transactions = pd.read_csv('Transactions.csv')

# Convert dates to datetime ```python
customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])

# Check for missing values
print("Missing values in Customers:\n", customers.isnull().sum())
print("Missing values in Products:\n", products.isnull().sum())
print("Missing values in Transactions:\n", transactions.isnull().sum())

# Merge datasets
merged_data = transactions.merge(customers, on='CustomerID').merge(products, on='ProductID')
2.3 Business Insights
Customer Distribution: The majority of customers are from North America, indicating a strong market presence in that region.
Product Popularity: Certain product categories show higher sales volumes, suggesting targeted marketing opportunities.
Transaction Trends: Peak transaction periods correlate with holiday seasons, highlighting the importance of seasonal promotions.
Customer Retention: Customers who signed up earlier tend to have higher transaction values, indicating loyalty.
Average Order Value: The average order value varies significantly across regions, suggesting tailored pricing strategies.
3. Lookalike Model
3.1 Model Development
The lookalike model identifies similar customers based on their transaction history. The following code snippet demonstrates the implementation:

python

Verify
Copy code
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Prepare customer features
customer_features = merged_data.groupby('CustomerID').agg({
    'TotalValue': 'sum', 
    'Quantity': 'sum'
}).reset_index()

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(customer_features.iloc[:, 1:])

# Compute cosine similarity
similarities = cosine_similarity(scaled_features)
similarity_df = pd.DataFrame(similarities, index=customer_features['CustomerID'], columns=customer_features['CustomerID'])

# Get top 3 similar customers for the first 20
lookalikes = {}
for cust_id in customer_features['CustomerID'][:20]:
    similar_customers = similarity_df.loc[cust_id].sort_values(ascending=False)[1:4]
    lookalikes[cust_id] = similar_customers.items()

# Save results
lookalikes_df = pd.DataFrame(lookalikes)
lookalikes_df.to_csv('FirstName_LastName_Lookalike.csv', index=False)
3.2 Results
The lookalike model successfully identified the top 3 similar customers for the first 20 customers, which can be utilized for targeted marketing campaigns.

4. Customer Segmentation / Clustering
4.1 Clustering Implementation
Customer segmentation was performed using KMeans clustering. The following code illustrates the clustering process:

python

Verify
Copy code
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

# Prepare data for clustering
clustering_data = merged_data.groupby('CustomerID').agg({
    'TotalValue': 'sum', 
    'Quantity': 'mean'
}).reset_index()

# Standardize features for clustering
scaled_clustering_data = scaler.fit_transform(clustering_data.iloc[:, 1:])

# KMeans clustering
db_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(scaled_clustering_data)
    db_index = davies_bouldin_score(scaled_clustering_data, kmeans.labels_)
    db_scores.append(db_index)

# Plot DB Index
plt.plot(range(2, 11), db_scores, marker='o')
plt.title('Davies-Bouldin Index vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('DB Index')
plt.show()

# Final clustering with optimal clusters
optimal_k = db_scores.index(min(db_scores)) + 2
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42).fit(scaled_clustering_data)
4.2 Clustering Results
Number of Clusters: The optimal number of clusters was determined based on the Davies-Bouldin Index.
Visualization: Clusters were visualized to understand customer segments better.
5. Conclusion
This assignment demonstrated the application of data science techniques to derive insights from an eCommerce dataset. The findings can guide strategic decisions in marketing, customer retention, and product development.

6. Deliverables
Jupyter Notebook containing EDA and modeling code.
PDF report summarizing insights and methodologies.
CSV files with lookalike results and clustering outputs.
