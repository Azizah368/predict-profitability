import streamlit as st
import os
import pickle
from joblib import load
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack

# Periksa keberadaan file model
if not os.path.exists('gradient_boosting_model.pkl'):
    st.error("File model tidak ditemukan.")
    st.stop()  # Hentikan eksekusi aplikasi jika file model tidak ditemukan

# Coba memuat model terlatih
try:
    model = load('gradient_boosting_model.pkl')
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")
    st.stop()  # Hentikan eksekusi aplikasi jika terjadi kesalahan

# Muat TF-IDF dan Label Encoder
try:
    tfidf = load('tfidf_vectorizer.pkl')
    label_encoder_restaurant = load('label_encoder_restaurant.pkl')
    label_encoder_menu_category = load('label_encoder_menu_category.pkl')
    label_encoder_menu_item = load('label_encoder_menu_item.pkl')
    label_encoder_profitability = load('label_encoder_profitability.pkl')
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat encoder atau vectorizer: {e}")
    st.stop()  # Hentikan eksekusi aplikasi jika terjadi kesalahan

# Lanjutkan dengan bagian kode lainnya...
