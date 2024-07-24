import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack

# Memuat model terlatih
model = pickle.load(open('gradient_boosting_model.pkl', 'rb'))

# Memuat TF-IDF dan Label Encoder
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
label_encoder_restaurant = pickle.load(open('label_encoder_restaurant.pkl', 'rb'))
label_encoder_menu_category = pickle.load(open('label_encoder_menu_category.pkl', 'rb'))
label_encoder_menu_item = pickle.load(open('label_encoder_menu_item.pkl', 'rb'))
label_encoder_profitability = pickle.load(open('label_encoder_profitability.pkl', 'rb'))

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
