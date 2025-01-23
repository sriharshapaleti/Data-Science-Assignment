#load the csv files data

import pandas as pd
import matplotlib.pyplot as plt

customers=pd.read_csv("Customers.csv")
products=pd.read_csv("Products.csv")
transactions = pd.read_csv("Transactions.csv")


#print the data

print(transactions.head())
print(transactions.info())
print(transactions.describe())

transactions ['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'], errors='coerce')

invalid_dates = transactions[pd.to_datetime(transactions['TransactionDate'], errors='coerce').isna()]
print(invalid_dates)

print(transactions['TransactionDate'].dtype)

#checking for duplicates
#for customers
print(customers.duplicated().sum())

print(customers['SignupDate'].min(),customers['SignupDate'].max())


#for products
print(products[products['Price'] < 0])


#for transactions
invalid_transactions = transactions[transactions['TotalValue']!=transactions['Quantity']* transactions['Price']]
print(invalid_transactions)



# Convert TransactionDate to datetime if not already done transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])
# Group transactions by month/year
transactions ['YearMonth'] = transactions['TransactionDate'].dt.to_period('M')
transaction_counts = transactions['YearMonth'].value_counts().sort_index()

# Plot the trend
plt.figure(figsize=(10,6))
transaction_counts.plot(kind='line' )
plt.title('Monthly Transactions Over Time')
plt.xlabel('Year-Month')
plt.ylabel('Number of Transactions')
plt.show()


# Analyze region distribution
region_counts = customers['Region'].value_counts()
print(region_counts)
region_counts.plot(kind='bar', title='Customer Distribution by Region')
plt.show()

# Analyze signup trends
customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
signup_trends = customers['SignupDate'].dt.to_period('M').value_counts().sort_index()
signup_trends.plot(kind='line', title='Customer Signup Trends')
plt.show()

# Merging datasets
merged_data = transactions.merge(customers, on='CustomerID', how='left').merge(products, on='ProductID', how='left')

# Inspect the merged data
print(merged_data.info())
print(merged_data.head())

# Calculate total revenue by product category
category_revenue = merged_data.groupby('Category')['TotalValue'].sum().sort_values(ascending=False)
print(category_revenue)

# Plot the revenue contribution by category
category_revenue.plot(kind='bar', figsize=(10, 6), title='Revenue by Product Category')
plt.ylabel('Total Revenue (USD)')
plt.show()

# Calculate the most sold product in each category
top_products = merged_data.groupby(['Category', 'ProductName'])['Quantity'].sum().reset_index()
top_products = top_products.sort_values(['Category', 'Quantity'], ascending=[True, False]).groupby('Category').head(1)
print(top_products)


# Revenue by region
region_revenue = merged_data.groupby('Region')['TotalValue'].sum().sort_values(ascending=False)
print(region_revenue)

# Plot region-wise revenue distribution
region_revenue.plot(kind='pie', autopct='%1.1f%%', figsize=(8, 8), title='Revenue by Region')
plt.ylabel('')
plt.show()

# Calculate customer lifetime value
customer_ltv = merged_data.groupby('CustomerID')['TotalValue'].sum().sort_values(ascending=False)
print(customer_ltv.head(10))

# Plot top customers by total revenue
customer_ltv.head(10).plot(kind='bar', figsize=(10, 6), title='Top 10 Customers by Total Revenue')
plt.ylabel('Total Revenue (USD)')
plt.xlabel('CustomerID')
plt.show()


# Monthly revenue trends
merged_data['TransactionMonth'] = merged_data['TransactionDate'].dt.to_period('M')
monthly_revenue = merged_data.groupby('TransactionMonth')['TotalValue'].sum()

# Plot monthly revenue trends
monthly_revenue.plot(kind='line', figsize=(10, 6), title='Monthly Revenue Trend')
plt.ylabel('Revenue (USD)')
plt.xlabel('Month')
plt.show()


# Aggregating data for each customer
customer_features = merged_data.groupby('CustomerID').agg({
    'TotalValue': 'sum',
    'Quantity': 'sum',
    'Category': lambda x: x.mode()[0],  # Most frequent category
    'Region': 'first'}).reset_index()

# Encode categorical data (Region, Category)
customer_features = pd.get_dummies(customer_features, columns=['Region', 'Category'], drop_first=True)

# Normalize numerical features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
numerical_cols = ['TotalValue', 'Quantity']
customer_features[numerical_cols] = scaler.fit_transform(customer_features[numerical_cols])

print(customer_features.head())

from sklearn.metrics.pairwise import cosine_similarity

# Calculate similarity matrix
similarity_matrix = cosine_similarity(customer_features.drop(columns=['CustomerID']))
similarity_df = pd.DataFrame(similarity_matrix, index=customer_features['CustomerID'], columns=customer_features['CustomerID'])

# Find top 3 similar customers for first 20 customers
lookalike_results = {}
for customer_id in customer_features['CustomerID'][:20]:
    similar_customers = similarity_df[customer_id].nlargest(4).iloc[1:4]  # Exclude self
    lookalike_results[customer_id] = similar_customers

print(lookalike_results)

from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

# K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
customer_features['Cluster'] = kmeans.fit_predict(customer_features.drop(columns=['CustomerID']))

# Evaluate clustering with Davies-Bouldin Index
db_index = davies_bouldin_score(customer_features.drop(columns=['CustomerID', 'Cluster']), customer_features['Cluster'])
print(f"Davies-Bouldin Index: {db_index}")










