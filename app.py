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
    st.markdown('### Pilih ID Restoran')
    st.markdown('''
    - **0** = R001
    - **1** = R002
    - **2** = R003
    ''')
    restaurant_ids = [0, 1, 2]
    restaurant_id = st.selectbox('Select Restaurant ID', options=restaurant_ids)

    # Opsi untuk memilih Menu Category
    st.markdown('### Pilih Kategori Menu')
    st.markdown('Kategori menu yang mencakup berbagai jenis makanan:')
    st.markdown('''
    - **0** = Appetizers
    - **1** = Beverage
    - **2** = Desserts
    - **3** = Main Course
    ''')
    menu_categories = [0, 1, 2, 3]
    menu_category = st.selectbox('Select Menu Category', options=menu_categories)

    # Opsi untuk memilih Menu Item
    st.markdown('### Pilih Item Menu')
    st.markdown('Item spesifik dari menu:')
    st.markdown('''
    - **0** = Bruschetta
    - **1** = Caprese Salad
    - **2** = Chicken Alfredo
    - **3** = Chocolate Lava Cake
    - **4** = Coffee
    - **5** = Fruit Tart
    - **6** = Grilled Steak
    - **7** = Iced Tea
    - **8** = Lemonade
    - **9** = New York Cheesecake
    - **10** = Shrimp Scampi
    - **11** = Soda
    - **12** = Spinach Artichoke Dip
    - **13** = Stuffed Mushrooms
    - **14** = Tiramisu
    - **15** = Vegetable Stir-Fry
    ''')
    menu_items = list(range(16))  # 0 hingga 15
    menu_item = st.selectbox('Select Menu Item', options=menu_items)

    # Opsi untuk memilih Ingredients
    st.markdown('### Pilih Bahan')
    st.markdown('Bahan-bahan yang digunakan dalam menu. Pilih dari daftar bahan yang tersedia.')
    ingredients_options = [
        'confidential', 'Tomatoes', 'Basil', 'Garlic', 'Olive Oil', 'Chocolate',
        'Butter', 'Sugar', 'Eggs', 'Chicken', 'Fettuccine', 'Alfredo Sauce', 'Parmesan'
    ]
    ingredients_selected = st.multiselect('Select Ingredients', options=ingredients_options)

    # Input untuk Price
    st.markdown('### Masukkan Harga')
    st.markdown('Harga item menu')
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
        st.write(f'Probabilities for each class: {dict(zip(labels, proba))}')
