# drug_app.py
import streamlit as st
import pandas as pd

def main():
    st.title("🚀 Drug Interaction Classifier")
    st.write("Upload a CSV file to check for drug interactions")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose CSV", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File loaded successfully!")
        st.write("Preview:", df.head())
        
        # Add your ML classification logic here

if __name__ == "__main__":
    main()
