# AI-Powered Multi-Agent Recommendation System

## 1️Install Dependencies
```bash
pip install flask pandas numpy scikit-learn sqlite3
```

## 2️Backend: Flask API
```python
from flask import Flask, request, jsonify
import sqlite3
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def connect_db():
    return sqlite3.connect("recommendation_system.db")

# Load Customer Data
def load_customer_data():
    conn = connect_db()
    df = pd.read_sql_query("SELECT * FROM customers", conn)
    conn.close()
    return df

# Load Product Data
def load_product_data():
    conn = connect_db()
    df = pd.read_sql_query("SELECT * FROM products", conn)
    conn.close()
    return df

# AI Model - Collaborative Filtering
def recommend_products(customer_id):
    customers = load_customer_data()
    products = load_product_data()
    
    if customer_id not in customers["customer_id"].values:
        return ["No recommendations available"]
    
    user_data = customers[customers["customer_id"] == customer_id]
    user_purchases = user_data[["product_id", "purchase_count"]]
    
    product_matrix = products.pivot(index='product_id', columns='category', values='popularity').fillna(0)
    similarity = cosine_similarity(product_matrix)
    similar_products = similarity[user_purchases["product_id"].values].mean(axis=0)
    
    recommended_products = products.iloc[np.argsort(similar_products)[::-1]]["product_name"].head(5).tolist()
    return recommended_products

@app.route("/recommend", methods=["GET"])
def get_recommendations():
    customer_id = int(request.args.get("customer_id"))
    recommendations = recommend_products(customer_id)
    return jsonify({"customer_id": customer_id, "recommendations": recommendations})

if __name__ == "__main__":
    app.run(debug=True)
```

## 3️SQLite Database Setup
```sql
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT,
    product_id INTEGER,
    purchase_count INTEGER
);

CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT,
    category TEXT,
    popularity FLOAT
);
```

## 4 Web App Frontend (HTML + JavaScript)
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <title>AI Product Recommendation</title>
</head>
<body>
    <h2>Get Product Recommendations</h2>
    <input type="number" id="customer_id" placeholder="Enter Customer ID">
    <button onclick="fetchRecommendations()">Get Recommendations</button>
    <ul id="recommendations"></ul>
    
    <script>
        function fetchRecommendations() {
            let customerId = document.getElementById("customer_id").value;
            fetch(`http://127.0.0.1:5000/recommend?customer_id=${customerId}`)
                .then(response => response.json())
                .then(data => {
                    let recList = document.getElementById("recommendations");
                    recList.innerHTML = "";
                    data.recommendations.forEach(item => {
                        let li = document.createElement("li");
                        li.textContent = item;
                        recList.appendChild(li);
                    });
                });
        }
    </script>
</body>
</html>
```

