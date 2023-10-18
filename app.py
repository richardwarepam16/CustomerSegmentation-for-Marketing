import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Function to preprocess data
def preprocess_data(df):
    df = df.dropna(subset=['Description', 'CustomerID'])
    df = df.drop_duplicates()
    return df

# Function to calculate RFM metrics
def calculate_rfm(df):
    df['MonetaryValue'] = df['Quantity'] * df['UnitPrice']
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (df['InvoiceDate'].max() - x.max()).days,
        'InvoiceNo': 'nunique',
        'MonetaryValue': 'sum'
    })
    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'MonetaryValue': 'MonetaryValue'
    }, inplace=True)
    return rfm

# Main Streamlit App
st.title('Customer Segmentation using K-Means')

# Upload Data
uploaded_file = st.file_uploader("Upload your dataset", type=["xlsx", "csv"])
if uploaded_file is not None:
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    
    st.write("Data Preview:")
    st.dataframe(df.head())
    # Data Preprocessing
    st.subheader('Data Preprocessing')
    if df is not None and not df.empty:
        df_cleaned = preprocess_data(df)
        st.write("Cleaned Data Preview:")
        st.dataframe(df_cleaned.head())

        # RFM Calculation
        st.subheader('RFM Calculation')
        rfm_df = calculate_rfm(df_cleaned)
        if rfm_df is not None and not rfm_df.empty:
            st.write("RFM Data Preview:")
            st.dataframe(rfm_df.head())
    
            # Data Normalization
            st.subheader('Data Normalization')
            scaler = StandardScaler()
            rfm_normalized = scaler.fit_transform(rfm_df)
            if rfm_normalized is not None:
                st.write("Normalized RFM Data Preview:")
                st.dataframe(pd.DataFrame(rfm_normalized, columns=rfm_df.columns).head())
        
                # K-Means Clustering
                st.subheader('K-Means Clustering')
                n_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)
                kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                kmeans.fit(rfm_normalized)  # Make sure rfm_normalized is not None or empty
                rfm_df['Cluster'] = kmeans.labels_
                st.write("Data with Cluster Labels:")
                st.dataframe(rfm_df.head())


                # Cluster Analysis (Optional)
                cluster_summary = rfm_df.groupby('Cluster').agg({
                    'Recency': 'mean',
                    'Frequency': 'mean',
                    'MonetaryValue': ['mean', 'count']
                }).reset_index()
                cluster_summary.columns = ['Cluster', 'Avg_Recency', 'Avg_Frequency', 'Avg_MonetaryValue', 'Customer_Count']
                st.subheader('Cluster Analysis')
                st.table(cluster_summary)
