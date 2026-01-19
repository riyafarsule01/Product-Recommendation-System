import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load Dataset
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_csv("2020-Jan-cleaned-sample.csv")
    return data

# Preprocess purchases and create product lookup
def preprocess_data(data):
    purchases = data[data['event_type'] == 'purchase']
    product_lookup = purchases.groupby('product_id').agg({
        'brand': 'first',
        'category_code': 'first',
        'price': 'mean'
    }).reset_index()
    return purchases, product_lookup

# Create User-Item Matrix
def create_user_item_matrix(purchases):
    user_item = purchases.pivot_table(index='user_id', columns='product_id', values='event_type', aggfunc='count')
    return user_item.fillna(0)

# Generate Recommendations
def get_recommendations(user_id, user_item_matrix, top_n=5):
    if user_id not in user_item_matrix.index:
        return []

    similarity_matrix = cosine_similarity(user_item_matrix)
    similarity_df = pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

    similar_users = similarity_df[user_id].sort_values(ascending=False)[1:]
    top_similar_users = similar_users.head(5).index

    recommendations = user_item_matrix.loc[top_similar_users].sum().sort_values(ascending=False)
    user_history = user_item_matrix.loc[user_id]
    recommendations = recommendations[user_history == 0]
    return recommendations.head(top_n).index.tolist()

# Show product info
def display_product_info(pid, lookup):
    product_info = lookup[lookup['product_id'] == pid]
    if not product_info.empty:
        row = product_info.iloc[0]
        st.markdown(f"""
        - *Product ID:* {pid}  
          *Brand:* {row['brand'] if pd.notna(row['brand']) else 'Unknown'}  
          *Category:* {row['category_code'] if pd.notna(row['category_code']) else 'Unknown'}  
          *Avg Price:* ₹{round(row['price'], 2) if pd.notna(row['price']) else 'N/A'}
        """)
    else:
        st.markdown(f"- *Product ID:* {pid} (Details not available)")

# Main App
def main():
    st.set_page_config(page_title="Suggestly", layout="wide")
    st.title("🛍 SUGGESTLY - Personalized Product Recommendations")
    st.write("Get product suggestions based on real purchase behavior and similar user interests.")

    # Upload section
    uploaded_file = st.file_uploader("📤 Upload your dataset (CSV)", type=['csv'])

    MAX_FILE_SIZE_MB = 5000  # 5GB limit
    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error("❌ File size exceeds the 5GB limit. Please upload a smaller file.")
            return
        data = load_data(uploaded_file)
    else:
        data = load_data()

    # Preprocess
    purchases, product_lookup = preprocess_data(data)
    user_item_matrix = create_user_item_matrix(purchases)
    users = user_item_matrix.index.tolist()

    # Sidebar filters
    st.sidebar.header("🔧 Filters & Tools")
    user_id = st.sidebar.selectbox("Select User ID", users)

    # Price range filter
    min_price = int(product_lookup['price'].min())
    max_price = int(product_lookup['price'].max())
    selected_price = st.sidebar.slider("💸 Price Range", min_price, max_price, (min_price, max_price))

    # Brand filter
    brands = ["All"] + sorted(product_lookup['brand'].dropna().unique().tolist())
    selected_brand = st.sidebar.selectbox("🏷 Filter by Brand", brands)

    # Generate recommendations
    if st.sidebar.button("🔎 Get Recommendations"):
        recommended_products = get_recommendations(user_id, user_item_matrix)

        st.subheader("🎯 Recommended Products for You:")
        filtered_recs = []
        for pid in recommended_products:
            row = product_lookup[product_lookup['product_id'] == pid]
            if not row.empty:
                price = row['price'].values[0]
                brand = row['brand'].values[0]
                if selected_price[0] <= price <= selected_price[1] and (selected_brand == "All" or brand == selected_brand):
                    filtered_recs.append(pid)

        if not filtered_recs:
            st.warning("No recommendations match your filter criteria.")
        else:
            for pid in filtered_recs:
                display_product_info(pid, product_lookup)

    # Product Explorer
    st.markdown("---")
    st.subheader("🔍 Explore Products")
    search_term = st.text_input("Search by Brand or Category:")
    filtered_products = product_lookup[
        product_lookup['brand'].str.contains(search_term, na=False, case=False) |
        product_lookup['category_code'].str.contains(search_term, na=False, case=False)
    ]
    st.write(f"Found *{len(filtered_products)}* matching products:")
    st.dataframe(filtered_products[['product_id', 'brand', 'category_code', 'price']])

    # Visual Analytics
    st.markdown("---")
    st.subheader("📈 User-Product Interaction Analytics")

    top_products = purchases['product_id'].value_counts().head(10)
    st.line_chart(top_products.rename("Purchases"))
    st.caption("Top 10 Most Purchased Products (Line Chart)")

    top_brands = purchases['brand'].value_counts().dropna().head(10)
    st.bar_chart(top_brands.rename("Purchases"))
    st.caption("Top 10 Brands (Bar Chart)")

    category_dist = purchases['category_code'].value_counts().dropna().head(10)
    st.pyplot(category_dist.plot.pie(autopct='%1.1f%%', figsize=(5, 5)).get_figure())
    st.caption("Category-wise Purchase Count (Pie Chart)")

    st.markdown("---")
    st.caption("🚀 Suggestly  — powered by Streamlit.")

if __name__ == "__main__":
    main()