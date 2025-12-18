# ============================================
# 🤖 MAKİNE ÖĞRENMESİ MODELLERİ MODÜLÜ
# ============================================
# 🎯 AMAÇ: Film/dizi öneri sistemi için farklı ML algoritmalarını uygular
# 📝 AÇIKLAMA: Bu modül 6 farklı ML yöntemi içerir:
#              1. TF-IDF + Kosinüs Benzerliği (İçerik Tabanlı)
#              2. K-En Yakın Komşu (KNN)
#              3. Random Forest
#              4. Lineer Regresyon / Ridge
#              5. SVD (Tekillik Ayrışımı)
#              6. Sinir Ağı (MLP)
# ============================================

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score
import warnings
import time
warnings.filterwarnings('ignore')


# ============================================
# 🔧 YARDIMCI FONKSİYONLAR
# ============================================

def combine_features(row):
    """
    🎯 AMAÇ: Birden fazla özelliği tek bir metin olarak birleştirir
    📊 KULLANIM: TF-IDF vektörizasyonu için
    💡 ÖRNEĞİN: "Action Drama | Christopher Nolan | Leonardo DiCaprio"
    """
    features = []
    
    # Türler
    if 'genres_str' in row.index and pd.notna(row['genres_str']):
        features.append(str(row['genres_str']))
    
    # Anahtar kelimeler
    if 'keywords_str' in row.index and pd.notna(row['keywords_str']):
        features.append(str(row['keywords_str']))
    
    # Yönetmen
    if 'director' in row.index and pd.notna(row['director']):
        features.append(str(row['director']))
    
    # Oyuncular
    if 'cast_str' in row.index and pd.notna(row['cast_str']):
        features.append(str(row['cast_str']))
    
    # Özet
    if 'overview' in row.index and pd.notna(row['overview']):
        features.append(str(row['overview']))
    
    return ' '.join(features)


# ============================================
# 📊 MODEL 1: İÇERİK TABANLI FİLTRELEME
# (TF-IDF + Kosinüs Benzerliği)
# ============================================

class ContentBasedRecommender:
    """
    🎯 AMAÇ: Metin benzerliğine dayalı öneri sistemi
    
    📊 YÖNTEM:
    1. TF-IDF (Term Frequency-Inverse Document Frequency) ile metin vektörizasyonu
    2. Kosinüs benzerliği ile filmler arası benzerlik hesabı
    
    💡 NASIL ÇALIŞIR:
    - Her filmin özet, tür, oyuncu bilgilerini birleştirir
    - Bu metinleri sayısal vektörlere dönüştürür
    - Vektörler arası açıyı (benzerliği) hesaplar
    - En benzer filmleri önerir
    """
    
    def __init__(self, max_features=5000):
        """
        🔧 BAŞLATICI
        📊 PARAMETRE:
        - max_features: TF-IDF'de kullanılacak maksimum kelime sayısı
        """
        self.name = "İçerik Tabanlı (TF-IDF)"
        self.max_features = max_features
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',  # İngilizce duraksama kelimeleri çıkar
            ngram_range=(1, 2)     # Tek kelime ve ikili kelime grupları
        )
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.df = None
        self.indices = None
        self.fit_time = 0
        
    def fit(self, df):
        """
        🎓 MODEL EĞİTİMİ
        🎯 AMAÇ: TF-IDF matrisini ve benzerlik matrisini hesaplar
        
        📊 ADIMLAR:
        1. Özellikleri birleştir
        2. TF-IDF vektörizasyonu uygula
        3. Kosinüs benzerlik matrisini hesapla
        """
        start_time = time.time()
        
        self.df = df.copy().reset_index(drop=True)
        
        # Özellikleri birleştir
        self.df['combined_features'] = self.df.apply(combine_features, axis=1)
        
        # TF-IDF vektörizasyonu
        # 💡 Bu adım her kelimeyi bir sayıya dönüştürür
        # Nadir kelimeler daha yüksek ağırlık alır
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['combined_features'])
        
        # Kosinüs benzerliği hesapla
        # 💡 Her film çifti için 0-1 arası benzerlik skoru
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        # Film başlığı -> index eşlemesi
        self.indices = pd.Series(self.df.index, index=self.df['title']).drop_duplicates()
        
        self.fit_time = time.time() - start_time
        
        return self
    
    def recommend(self, title, n=10):
        """
        🎬 ÖNERİ ÜRET
        🎯 AMAÇ: Verilen filme benzer filmleri döndürür
        
        📊 ADIMLAR:
        1. Filmin index'ini bul
        2. Benzerlik skorlarını al
        3. En yüksek skorlu filmleri seç (kendisi hariç)
        """
        if title not in self.indices:
            return pd.DataFrame()
        
        idx = self.indices[title]
        
        # Benzerlik skorlarını al
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        
        # Skora göre sırala (azalan)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # İlk n+1 sonucu al (kendisi dahil)
        sim_scores = sim_scores[1:n+1]
        
        # Film indekslerini al
        movie_indices = [i[0] for i in sim_scores]
        
        # Sonuçları DataFrame olarak döndür
        result = self.df.iloc[movie_indices][['title', 'genres_str', 'vote_average']].copy()
        result['similarity_score'] = [s[1] for s in sim_scores]
        
        return result


# ============================================
# 📊 MODEL 2: K-EN YAKIN KOMŞU (KNN)
# ============================================

class KNNRecommender:
    """
    🎯 AMAÇ: Feature vektörleri ile en yakın komşuları bulur
    
    📊 YÖNTEM:
    - TF-IDF vektörlerini kullanır
    - K-NN algoritması ile en yakın N filmi bulur
    
    💡 AVANTAJI:
    - Hızlı ve basit
    - Yeni içerikler için anında çalışır
    """
    
    def __init__(self, n_neighbors=10, metric='cosine'):
        """
        🔧 BAŞLATICI
        📊 PARAMETRELER:
        - n_neighbors: Kaç komşu bulunacak
        - metric: Mesafe ölçütü (cosine, euclidean, manhattan)
        """
        self.name = "K-En Yakın Komşu (KNN)"
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.model = NearestNeighbors(n_neighbors=n_neighbors+1, metric=metric)
        self.tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
        self.tfidf_matrix = None
        self.df = None
        self.fit_time = 0
        
    def fit(self, df):
        """
        🎓 MODEL EĞİTİMİ
        """
        start_time = time.time()
        
        self.df = df.copy().reset_index(drop=True)
        self.df['combined_features'] = self.df.apply(combine_features, axis=1)
        
        # TF-IDF vektörizasyonu
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['combined_features'])
        
        # KNN modelini eğit
        self.model.fit(self.tfidf_matrix)
        
        self.fit_time = time.time() - start_time
        
        return self
    
    def recommend(self, title, n=10):
        """
        🎬 ÖNERİ ÜRET
        """
        if title not in self.df['title'].values:
            return pd.DataFrame()
        
        idx = self.df[self.df['title'] == title].index[0]
        
        # En yakın komşuları bul
        distances, indices = self.model.kneighbors(self.tfidf_matrix[idx])
        
        # Kendisi hariç sonuçları al
        movie_indices = indices.flatten()[1:n+1]
        distance_scores = distances.flatten()[1:n+1]
        
        result = self.df.iloc[movie_indices][['title', 'genres_str', 'vote_average']].copy()
        result['similarity_score'] = 1 - distance_scores  # Mesafeyi benzerliğe çevir
        
        return result


# ============================================
# 📊 MODEL 3: RANDOM FOREST
# ============================================

class RandomForestRecommender:
    """
    🎯 AMAÇ: Ağaç tabanlı sınıflandırma ile öneri
    
    📊 YÖNTEM:
    - Türleri hedef değişken olarak kullanır
    - Random Forest ile tür tahmini yapar
    - Benzer türdeki filmleri önerir
    
    💡 AVANTAJI:
    - Özellik önemini görebiliriz
    - Kategori ve sayısal verileri birlikte kullanabilir
    """
    
    def __init__(self, n_estimators=100, max_depth=10):
        """
        🔧 BAŞLATICI
        📊 PARAMETRELER:
        - n_estimators: Ağaç sayısı
        - max_depth: Maksimum ağaç derinliği
        """
        self.name = "Random Forest"
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1  # Tüm CPU çekirdeklerini kullan
        )
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = StandardScaler()
        self.mlb = MultiLabelBinarizer()
        self.df = None
        self.feature_matrix = None
        self.fit_time = 0
        self.feature_importance = None
        
    def fit(self, df):
        """
        🎓 MODEL EĞİTİMİ
        """
        start_time = time.time()
        
        self.df = df.copy().reset_index(drop=True)
        
        # Metin özelliklerini vektörize et
        self.df['combined_features'] = self.df.apply(combine_features, axis=1)
        tfidf_features = self.tfidf.fit_transform(self.df['combined_features']).toarray()
        
        # Sayısal özellikleri normalize et
        numeric_cols = ['popularity', 'vote_average', 'vote_count']
        available_numeric = [c for c in numeric_cols if c in self.df.columns]
        
        if available_numeric:
            numeric_features = self.scaler.fit_transform(
                self.df[available_numeric].fillna(0)
            )
            # Özellikleri birleştir
            self.feature_matrix = np.hstack([tfidf_features, numeric_features])
        else:
            self.feature_matrix = tfidf_features
        
        # Hedef değişken olarak vote_average kullan
        y = self.df['vote_average'].fillna(self.df['vote_average'].mean())
        
        # Modeli eğit
        self.model.fit(self.feature_matrix, y)
        
        # Özellik önemlerini kaydet
        self.feature_importance = self.model.feature_importances_
        
        self.fit_time = time.time() - start_time
        
        return self
    
    def recommend(self, title, n=10):
        """
        🎬 ÖNERİ ÜRET
        - Hedef filmin özelliklerine en benzer filmleri bul
        """
        if title not in self.df['title'].values:
            return pd.DataFrame()
        
        idx = self.df[self.df['title'] == title].index[0]
        target_features = self.feature_matrix[idx].reshape(1, -1)
        
        # Tüm filmler için tahmin yap
        predictions = self.model.predict(self.feature_matrix)
        target_pred = predictions[idx]
        
        # Tahmin farkına göre sırala (benzer tahminler = benzer filmler)
        diffs = np.abs(predictions - target_pred)
        similar_indices = np.argsort(diffs)[1:n+1]
        
        result = self.df.iloc[similar_indices][['title', 'genres_str', 'vote_average']].copy()
        result['similarity_score'] = 1 - (diffs[similar_indices] / diffs.max())
        
        return result
    
    def get_feature_importance(self, top_n=20):
        """
        📊 ÖZELLİK ÖNEMLERİNİ DÖNDÜR
        💡 AÇIKLAMA: Hangi özellikler modelin kararlarını en çok etkiliyor
        """
        if self.feature_importance is None:
            return None
        
        # TF-IDF kelimelerinin isimlerini al
        feature_names = list(self.tfidf.get_feature_names_out())
        feature_names.extend(['popularity', 'vote_average', 'vote_count'])
        
        importance_df = pd.DataFrame({
            'feature': feature_names[:len(self.feature_importance)],
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)


# ============================================
# 📊 MODEL 4: LİNEER REGRESYON
# ============================================

class LinearRecommender:
    """
    🎯 AMAÇ: Lineer model ile puan tahmini
    
    📊 YÖNTEM:
    - Ridge regresyon kullanır (overfitting'e karşı regularizasyon)
    - Özelliklere göre film puanı tahmin eder
    - Benzer tahminli filmleri önerir
    
    💡 AVANTAJI:
    - Yorumlanabilirlik yüksek
    - Eğitim süresi kısa
    """
    
    def __init__(self, alpha=1.0):
        """
        🔧 BAŞLATICI
        📊 PARAMETRE:
        - alpha: Regularizasyon gücü
        """
        self.name = "Lineer Regresyon (Ridge)"
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.tfidf = TfidfVectorizer(max_features=2000, stop_words='english')
        self.scaler = StandardScaler()
        self.df = None
        self.feature_matrix = None
        self.fit_time = 0
        self.coefficients = None
        
    def fit(self, df):
        """
        🎓 MODEL EĞİTİMİ
        """
        start_time = time.time()
        
        self.df = df.copy().reset_index(drop=True)
        
        # Özellikleri hazırla
        self.df['combined_features'] = self.df.apply(combine_features, axis=1)
        tfidf_features = self.tfidf.fit_transform(self.df['combined_features']).toarray()
        
        # Sayısal özellikler
        numeric_cols = ['popularity', 'vote_count']
        available_numeric = [c for c in numeric_cols if c in self.df.columns]
        
        if available_numeric:
            numeric_features = self.scaler.fit_transform(
                self.df[available_numeric].fillna(0)
            )
            self.feature_matrix = np.hstack([tfidf_features, numeric_features])
        else:
            self.feature_matrix = tfidf_features
        
        # Hedef: vote_average
        y = self.df['vote_average'].fillna(self.df['vote_average'].mean())
        
        # Modeli eğit
        self.model.fit(self.feature_matrix, y)
        self.coefficients = self.model.coef_
        
        self.fit_time = time.time() - start_time
        
        return self
    
    def recommend(self, title, n=10):
        """
        🎬 ÖNERİ ÜRET
        """
        if title not in self.df['title'].values:
            return pd.DataFrame()
        
        idx = self.df[self.df['title'] == title].index[0]
        
        # Tahminler
        predictions = self.model.predict(self.feature_matrix)
        target_pred = predictions[idx]
        
        # Benzer tahminli filmleri bul
        diffs = np.abs(predictions - target_pred)
        similar_indices = np.argsort(diffs)[1:n+1]
        
        result = self.df.iloc[similar_indices][['title', 'genres_str', 'vote_average']].copy()
        result['similarity_score'] = 1 - (diffs[similar_indices] / (diffs.max() + 1e-10))
        
        return result


# ============================================
# 📊 MODEL 5: SVD (TEKİLLİK AYRIŞIMI)
# ============================================

class SVDRecommender:
    """
    🎯 AMAÇ: Matris faktörizasyonu ile boyut indirgeme ve öneri
    
    📊 YÖNTEM:
    - TF-IDF matrisini düşük boyutlu uzaya indirger
    - Gizli faktörleri keşfeder
    - Benzer gizli faktörlü filmler önerilir
    
    💡 AVANTAJI:
    - Büyük veri setlerinde etkili
    - Gürültüyü azaltır
    - "Latent features" yakalar
    """
    
    def __init__(self, n_components=100):
        """
        🔧 BAŞLATICI
        📊 PARAMETRE:
        - n_components: Gizli faktör sayısı
        """
        self.name = "SVD (Matris Faktörizasyonu)"
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
        self.df = None
        self.svd_matrix = None
        self.fit_time = 0
        
    def fit(self, df):
        """
        🎓 MODEL EĞİTİMİ
        """
        start_time = time.time()
        
        self.df = df.copy().reset_index(drop=True)
        
        # TF-IDF
        self.df['combined_features'] = self.df.apply(combine_features, axis=1)
        tfidf_matrix = self.tfidf.fit_transform(self.df['combined_features'])
        
        # SVD ile boyut indirgeme
        # 💡 3000 boyutlu vektörü 100 boyuta indirger
        self.svd_matrix = self.svd.fit_transform(tfidf_matrix)
        
        self.fit_time = time.time() - start_time
        
        return self
    
    def recommend(self, title, n=10):
        """
        🎬 ÖNERİ ÜRET
        """
        if title not in self.df['title'].values:
            return pd.DataFrame()
        
        idx = self.df[self.df['title'] == title].index[0]
        
        # SVD uzayında kosinüs benzerliği
        target_vector = self.svd_matrix[idx].reshape(1, -1)
        similarities = cosine_similarity(target_vector, self.svd_matrix).flatten()
        
        # En benzer filmleri bul
        similar_indices = similarities.argsort()[::-1][1:n+1]
        
        result = self.df.iloc[similar_indices][['title', 'genres_str', 'vote_average']].copy()
        result['similarity_score'] = similarities[similar_indices]
        
        return result
    
    def get_explained_variance(self):
        """
        📊 AÇIKLANAN VARYANS ORANI
        💡 AÇIKLAMA: SVD'nin ne kadar bilgiyi koruduğunu gösterir
        """
        return self.svd.explained_variance_ratio_.sum()


# ============================================
# 📊 MODEL 6: SİNİR AĞI (MLP)
# ============================================

class NeuralRecommender:
    """
    🎯 AMAÇ: Derin öğrenme ile öneri sistemi
    
    📊 YÖNTEM:
    - Multi-Layer Perceptron (MLP) kullanır
    - Gizli katmanlarla karmaşık örüntüleri öğrenir
    - Puan tahmini yapar
    
    💡 AVANTAJI:
    - Karmaşık ilişkileri yakalayabilir
    - Non-linear patterns öğrenebilir
    """
    
    def __init__(self, hidden_layers=(256, 128, 64)):
        """
        🔧 BAŞLATICI
        📊 PARAMETRE:
        - hidden_layers: Gizli katman boyutları
        """
        self.name = "Sinir Ağı (MLP)"
        self.hidden_layers = hidden_layers
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation='relu',      # ReLU aktivasyon fonksiyonu
            solver='adam',          # Adam optimizer
            max_iter=200,           # Maksimum iterasyon
            random_state=42,
            early_stopping=True,    # Erken durdurma (overfitting önleme)
            validation_fraction=0.1
        )
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = StandardScaler()
        self.df = None
        self.feature_matrix = None
        self.fit_time = 0
        
    def fit(self, df):
        """
        🎓 MODEL EĞİTİMİ
        """
        start_time = time.time()
        
        self.df = df.copy().reset_index(drop=True)
        
        # Özellikleri hazırla
        self.df['combined_features'] = self.df.apply(combine_features, axis=1)
        tfidf_features = self.tfidf.fit_transform(self.df['combined_features']).toarray()
        
        # Normalize et (Neural Networks için önemli!)
        self.feature_matrix = self.scaler.fit_transform(tfidf_features)
        
        # Hedef
        y = self.df['vote_average'].fillna(self.df['vote_average'].mean())
        
        # Modeli eğit
        self.model.fit(self.feature_matrix, y)
        
        self.fit_time = time.time() - start_time
        
        return self
    
    def recommend(self, title, n=10):
        """
        🎬 ÖNERİ ÜRET
        """
        if title not in self.df['title'].values:
            return pd.DataFrame()
        
        idx = self.df[self.df['title'] == title].index[0]
        
        # Tahminler
        predictions = self.model.predict(self.feature_matrix)
        target_pred = predictions[idx]
        
        # Benzer tahminli filmler
        diffs = np.abs(predictions - target_pred)
        similar_indices = np.argsort(diffs)[1:n+1]
        
        result = self.df.iloc[similar_indices][['title', 'genres_str', 'vote_average']].copy()
        result['similarity_score'] = 1 - (diffs[similar_indices] / (diffs.max() + 1e-10))
        
        return result


# ============================================
# 🏭 MODEL FABRİKASI
# ============================================

def get_all_models():
    """
    📦 TÜM MODELLERİ DÖNDÜR
    🎯 AMAÇ: Kullanılabilir tüm öneri modellerini listeler
    """
    return {
        'content_based': ContentBasedRecommender(),
        'knn': KNNRecommender(),
        'random_forest': RandomForestRecommender(),
        'linear': LinearRecommender(),
        'svd': SVDRecommender(),
        'neural': NeuralRecommender()
    }


# ============================================
# 🧪 TEST KODU
# ============================================

if __name__ == "__main__":
    from data_analysis import DataAnalyzer
    
    print("=" * 60)
    print("🤖 Makine Öğrenmesi Modelleri Test")
    print("=" * 60)
    
    # Veri yükle
    analyzer = DataAnalyzer()
    analyzer.load_data()
    analyzer.preprocess_movies()
    
    # Modelleri test et
    models = get_all_models()
    
    test_movie = "The Dark Knight"
    
    for name, model in models.items():
        print(f"\n{'='*40}")
        print(f"📊 Model: {model.name}")
        print(f"{'='*40}")
        
        # Eğit
        model.fit(analyzer.movies.head(1000))  # İlk 1000 film ile test
        print(f"   ⏱️ Eğitim süresi: {model.fit_time:.2f} saniye")
        
        # Öneri al
        recommendations = model.recommend(test_movie, n=5)
        
        if not recommendations.empty:
            print(f"\n   🎬 '{test_movie}' için öneriler:")
            for _, row in recommendations.iterrows():
                print(f"      - {row['title']} (⭐ {row['vote_average']:.1f})")
        else:
            print(f"   ⚠️ Film bulunamadı: {test_movie}")
    
    print("\n" + "=" * 60)
    print("✅ Tüm modeller başarıyla test edildi!")
    print("=" * 60)
