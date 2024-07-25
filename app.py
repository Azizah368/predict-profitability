import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
import os

# Periksa keberadaan file model
if not os.path.exists('gradient_boosting_model.joblib'):
    st.error("File model tidak ditemukan.")
else:
    # Memuat model terlatih
    model = joblib.load('gradient_boosting_model.joblib')

    # Memuat TF-IDF dan Label Encoder
    tfidf = joblib.load('tfidf_vectorizer.joblib')
    label_encoder_restaurant = joblib.load('label_encoder_restaurant.joblib')
    label_encoder_menu_category = joblib.load('label_encoder_menu_category.joblib')
    label_encoder_menu_item = joblib.load('label_encoder_menu_item.joblib')
    label_encoder_profitability = joblib.load('label_encoder_profitability.joblib')

    st.title('Prediksi Profitabilitas Menu Restoran')

    # Opsi untuk memilih Restaurant ID
    restaurant_ids = [0, 1, 2]
    restaurant_id = st.selectbox('Select Restaurant ID', options=restaurant_ids)

    # Opsi untuk memilih Menu Category
    menu_categories = [0, 1, 2, 3]
    menu_category = st.selectbox('Select Menu Category', options=menu_categories)

    # Opsi untuk memilih Menu Item
    menu_items = list(range(16))  # 0 hingga 15
    menu_item = st.selectbox('Select Menu Item', options=menu_items)

    # Opsi untuk memilih Ingredients
    ingredients_options = [
        'confidential', 'Tomatoes', 'Basil', 'Garlic', 'Olive Oil', 'Chocolate',
        'Butter', 'Sugar', 'Eggs', 'Chicken', 'Fettuccine', 'Alfredo Sauce', 'Parmesan'
    ]
    ingredients_selected = st.multiselect('Select Ingredients', options=ingredients_options)

    # Input untuk Price
    price = st.number_input('Price', min_value=0.0, step=0.01)

    if st.button('Predict'):
        # Menyiapkan fitur untuk prediksi
        restaurant_id_encoded = label_encoder_restaurant.transform([restaurant_id])[0]
        menu_category_encoded = label_encoder_menu_category.transform([menu_category])[0]
        menu_item_encoded = label_encoder_menu_item.transform([menu_item])[0]
        
        # Menyiapkan TF-IDF untuk Ingredients
        ingredients_combined = ', '.join(ingredients_selected)
        ingredients_tfidf = tfidf.transform([ingredients_combined])
        
        # Menyiapkan fitur untuk model
        features = hstack([
            pd.DataFrame([[restaurant_id_encoded, menu_category_encoded, menu_item_encoded, price]],
                         columns=['RestaurantID', 'MenuCategory', 'MenuItem', 'Price']).values,
            ingredients_tfidf
        ])
        
        # Melakukan prediksi
        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features)
        
        # Mengambil label dari probabilitas
        predicted_class = prediction[0]
        proba = prediction_proba[0]
        
        # Menyusun label berdasarkan encoding
        labels = label_encoder_profitability.classes_
        
        # Menampilkan hasil
        st.write(f'The predicted profitability is: {labels[predicted_class]}')
