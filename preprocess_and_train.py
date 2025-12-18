# ============================================
# 🔧 ÖN İŞLEME VE MODEL EĞİTİMİ SCRİPTİ
# ============================================
# 🎯 AMAÇ: Verileri bir kez işle, modelleri bir kez eğit ve kaydet
# 📝 KULLANIM: python preprocess_and_train.py
# 💡 NOT: Bu script sadece bir kez çalıştırılır, sonra app.py
#         kaydedilmiş verileri ve modelleri kullanır
# ============================================

import pandas as pd
import numpy as np
import pickle
import os
import ast
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🔧 VERİ ÖN İŞLEME VE MODEL EĞİTİMİ")
print("=" * 60)

# ============================================
# 📁 KLASÖR OLUŞTUR
# ============================================

MODELS_DIR = 'trained_models'
DATA_DIR = 'processed_data'

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================
# 🔧 YARDIMCI FONKSİYONLAR
# ============================================

def parse_json_column(data):
    try:
        if pd.isna(data):
            return []
        parsed = ast.literal_eval(str(data))
        if isinstance(parsed, list):
            return [item.get('name', '') if isinstance(item, dict) else str(item) for item in parsed]
        return []
    except:
        return []

def extract_director(crew_data):
    try:
        if pd.isna(crew_data):
            return ''
        crew = ast.literal_eval(str(crew_data))
        for member in crew:
            if member.get('job') == 'Director':
                return member.get('name', '')
        return ''
    except:
        return ''

def extract_top_cast(cast_data, n=3):
    try:
        if pd.isna(cast_data):
            return []
        cast = ast.literal_eval(str(cast_data))
        return [actor.get('name', '') for actor in cast[:n]]
    except:
        return []

def combine_features(row):
    features = []
    if 'genres_str' in row.index and pd.notna(row['genres_str']):
        features.append(str(row['genres_str']))
    if 'keywords_str' in row.index and pd.notna(row['keywords_str']):
        features.append(str(row['keywords_str']))
    if 'director' in row.index and pd.notna(row['director']):
        features.append(str(row['director']))
    if 'cast_str' in row.index and pd.notna(row['cast_str']):
        features.append(str(row['cast_str']))
    if 'overview' in row.index and pd.notna(row['overview']):
        # Özeti kısa tut (bellek için)
        overview = str(row['overview'])[:500]
        features.append(overview)
    return ' '.join(features)

# ============================================
# 📥 VERİ YÜKLEME VE İŞLEME
# ============================================

print("\n📥 Veriler yükleniyor...")

# Film verisi
movies_raw = pd.read_csv("tmdb_5000_movies.csv")
credits_raw = pd.read_csv("tmdb_5000_credits.csv")

print(f"   ✅ Filmler: {len(movies_raw):,} kayıt")

# Birleştir
movies = movies_raw.merge(credits_raw, left_on='id', right_on='movie_id', how='left', suffixes=('', '_credits'))

# Parse işlemleri
print("   🔄 Film özellikleri parse ediliyor...")
movies['genres_list'] = movies['genres'].apply(parse_json_column)
movies['genres_str'] = movies['genres_list'].apply(lambda x: ', '.join(x))
movies['keywords_list'] = movies['keywords'].apply(parse_json_column)
movies['keywords_str'] = movies['keywords_list'].apply(lambda x: ', '.join(x))
movies['director'] = movies['crew'].apply(extract_director)
movies['cast_list'] = movies['cast'].apply(lambda x: extract_top_cast(x, 3))
movies['cast_str'] = movies['cast_list'].apply(lambda x: ', '.join(x))
movies['overview'] = movies['overview'].fillna('')
movies['content_type'] = 'Film'

# Sadece gerekli sütunları tut (bellek tasarrufu)
movies = movies[['title', 'overview', 'genres_str', 'keywords_str', 'director', 
                  'cast_str', 'vote_average', 'vote_count', 'popularity', 'content_type']].copy()
movies = movies.dropna(subset=['title']).reset_index(drop=True)

print(f"   ✅ Film verisi hazırlandı: {len(movies):,} film")

# ============================================
# 📺 DİZİ VERİSİ - ÖRNEKLEM AL (PERFORMANS İÇİN)
# ============================================

print("\n📺 Dizi verisi yükleniyor (örneklem alınacak)...")

tv_raw = pd.read_csv("TMDB_tv_dataset_v3.csv")
print(f"   📊 Toplam dizi: {len(tv_raw):,}")

# Sadece popüler dizileri al (vote_count >= 10 ve vote_average > 0)
tv_filtered = tv_raw[(tv_raw['vote_count'] >= 10) & (tv_raw['vote_average'] > 0)].copy()
print(f"   📊 Filtrelenmiş dizi: {len(tv_filtered):,}")

# En popüler 5000 diziyi al (yeterli çeşitlilik için)
tv = tv_filtered.nlargest(5000, 'popularity').copy()
print(f"   📊 Örneklem: {len(tv):,} dizi")

# Sütun adlarını düzenle
tv = tv.rename(columns={'name': 'title'})

# Türleri parse et
def parse_tv_genres(genre_str):
    if pd.isna(genre_str):
        return ''
    return str(genre_str)

tv['genres_str'] = tv['genres'].apply(parse_tv_genres)
tv['keywords_str'] = ''
tv['director'] = tv['created_by'].fillna('')
tv['cast_str'] = ''
tv['overview'] = tv['overview'].fillna('')
tv['content_type'] = 'Dizi'

# Sadece gerekli sütunları tut
tv = tv[['title', 'overview', 'genres_str', 'keywords_str', 'director', 
         'cast_str', 'vote_average', 'vote_count', 'popularity', 'content_type']].copy()
tv = tv.dropna(subset=['title']).reset_index(drop=True)

print(f"   ✅ Dizi verisi hazırlandı: {len(tv):,} dizi")

# ============================================
# 💾 İŞLENMİŞ VERİLERİ KAYDET
# ============================================

print("\n💾 İşlenmiş veriler kaydediliyor...")

movies.to_pickle(f'{DATA_DIR}/movies_processed.pkl')
tv.to_pickle(f'{DATA_DIR}/tv_processed.pkl')

print(f"   ✅ {DATA_DIR}/movies_processed.pkl")
print(f"   ✅ {DATA_DIR}/tv_processed.pkl")

# ============================================
# 🤖 MODEL EĞİTİMİ
# ============================================

print("\n🤖 Modeller eğitiliyor...")

def train_and_save_models(df, content_type):
    """Tüm modelleri eğit ve kaydet"""
    
    print(f"\n   📊 {content_type} modelleri eğitiliyor ({len(df)} kayıt)...")
    
    # Combined features oluştur
    df = df.copy()
    df['combined_features'] = df.apply(combine_features, axis=1)
    
    models_data = {}
    
    # 1. İçerik Tabanlı (TF-IDF + Kosinüs)
    print(f"      1️⃣ İçerik Tabanlı (TF-IDF)...")
    start = time.time()
    tfidf = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    # Benzerlik matrisini HESAPLAMA - çok büyük olur, sadece TF-IDF kaydet
    models_data['content_based'] = {
        'tfidf': tfidf,
        'tfidf_matrix': tfidf_matrix,
        'titles': df['title'].tolist(),
        'df': df[['title', 'genres_str', 'vote_average', 'vote_count']].to_dict('records'),
        'fit_time': time.time() - start
    }
    print(f"         ⏱️ {models_data['content_based']['fit_time']:.2f}s")
    
    # 2. KNN
    print(f"      2️⃣ K-En Yakın Komşu (KNN)...")
    start = time.time()
    knn = NearestNeighbors(n_neighbors=11, metric='cosine')
    knn.fit(tfidf_matrix)
    models_data['knn'] = {
        'model': knn,
        'tfidf': tfidf,
        'tfidf_matrix': tfidf_matrix,
        'titles': df['title'].tolist(),
        'df': df[['title', 'genres_str', 'vote_average', 'vote_count']].to_dict('records'),
        'fit_time': time.time() - start
    }
    print(f"         ⏱️ {models_data['knn']['fit_time']:.2f}s")
    
    # 3. Random Forest (küçük veri ile)
    print(f"      3️⃣ Random Forest...")
    start = time.time()
    # Daha küçük TF-IDF
    tfidf_small = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf_small_matrix = tfidf_small.fit_transform(df['combined_features']).toarray()
    rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
    y = df['vote_average'].fillna(df['vote_average'].mean())
    rf.fit(tfidf_small_matrix, y)
    models_data['random_forest'] = {
        'model': rf,
        'tfidf': tfidf_small,
        'predictions': rf.predict(tfidf_small_matrix),
        'titles': df['title'].tolist(),
        'df': df[['title', 'genres_str', 'vote_average', 'vote_count']].to_dict('records'),
        'fit_time': time.time() - start
    }
    print(f"         ⏱️ {models_data['random_forest']['fit_time']:.2f}s")
    
    # 4. Lineer (Ridge)
    print(f"      4️⃣ Lineer Regresyon (Ridge)...")
    start = time.time()
    ridge = Ridge(alpha=1.0)
    ridge.fit(tfidf_small_matrix, y)
    models_data['linear'] = {
        'model': ridge,
        'tfidf': tfidf_small,
        'predictions': ridge.predict(tfidf_small_matrix),
        'titles': df['title'].tolist(),
        'df': df[['title', 'genres_str', 'vote_average', 'vote_count']].to_dict('records'),
        'fit_time': time.time() - start
    }
    print(f"         ⏱️ {models_data['linear']['fit_time']:.2f}s")
    
    # 5. SVD
    print(f"      5️⃣ SVD (Matris Faktörizasyonu)...")
    start = time.time()
    svd = TruncatedSVD(n_components=100, random_state=42)
    svd_matrix = svd.fit_transform(tfidf_matrix)
    models_data['svd'] = {
        'model': svd,
        'svd_matrix': svd_matrix,
        'titles': df['title'].tolist(),
        'df': df[['title', 'genres_str', 'vote_average', 'vote_count']].to_dict('records'),
        'fit_time': time.time() - start
    }
    print(f"         ⏱️ {models_data['svd']['fit_time']:.2f}s")
    
    # 6. Sinir Ağı (MLP)
    print(f"      6️⃣ Sinir Ağı (MLP)...")
    start = time.time()
    # Küçük MLP
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(tfidf_small_matrix)
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=100, random_state=42, early_stopping=True)
    mlp.fit(scaled_features, y)
    models_data['neural'] = {
        'model': mlp,
        'scaler': scaler,
        'tfidf': tfidf_small,
        'predictions': mlp.predict(scaled_features),
        'titles': df['title'].tolist(),
        'df': df[['title', 'genres_str', 'vote_average', 'vote_count']].to_dict('records'),
        'fit_time': time.time() - start
    }
    print(f"         ⏱️ {models_data['neural']['fit_time']:.2f}s")
    
    return models_data

# Film modelleri
film_models = train_and_save_models(movies, 'Film')
with open(f'{MODELS_DIR}/film_models.pkl', 'wb') as f:
    pickle.dump(film_models, f)
print(f"\n   ✅ {MODELS_DIR}/film_models.pkl kaydedildi")

# Dizi modelleri
tv_models = train_and_save_models(tv, 'Dizi')
with open(f'{MODELS_DIR}/tv_models.pkl', 'wb') as f:
    pickle.dump(tv_models, f)
print(f"   ✅ {MODELS_DIR}/tv_models.pkl kaydedildi")

# ============================================
# 📊 İSTATİSTİKLER
# ============================================

print("\n" + "=" * 60)
print("📊 ÖZET")
print("=" * 60)
print(f"   🎬 Film sayısı: {len(movies):,}")
print(f"   📺 Dizi sayısı: {len(tv):,}")
print(f"   🤖 Eğitilen model: 6 × 2 = 12")
print(f"\n   📁 Kaydedilen dosyalar:")
print(f"      - {DATA_DIR}/movies_processed.pkl")
print(f"      - {DATA_DIR}/tv_processed.pkl")
print(f"      - {MODELS_DIR}/film_models.pkl")
print(f"      - {MODELS_DIR}/tv_models.pkl")

print("\n✅ Ön işleme tamamlandı! Artık app.py çalıştırabilirsiniz.")
print("=" * 60)
