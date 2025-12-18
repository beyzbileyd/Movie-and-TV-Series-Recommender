# ============================================
# 📊 VERİ ANALİZİ MODÜLÜ
# ============================================
# 🎯 AMAÇ: TMDB veri setlerini analiz etmek ve görselleştirmek
# 📝 AÇIKLAMA: Bu modül film/dizi verilerinin istatistiksel analizini,
#              görselleştirmelerini ve özellik mühendisliğini sağlar.
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 🔧 YARDIMCI FONKSİYONLAR
# ============================================

def parse_json_column(data):
    """
    🎯 AMAÇ: JSON formatındaki sütunları Python listelerine çevirir
    📊 KULLANIM: genres, keywords, cast gibi sütunlar için
    💡 ÖRNEĞİN: "[{'id': 28, 'name': 'Action'}]" -> ['Action']
    """
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
    """
    🎯 AMAÇ: Ekip verisinden yönetmen ismini çıkarır
    📊 YÖNTEM: crew listesinde job='Director' olan kişiyi bulur
    """
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

def extract_top_cast(cast_data, n=5):
    """
    🎯 AMAÇ: İlk n oyuncuyu çıkarır
    📊 KULLANIM: Film afişinde görünen ana oyuncular
    """
    try:
        if pd.isna(cast_data):
            return []
        cast = ast.literal_eval(str(cast_data))
        return [actor.get('name', '') for actor in cast[:n]]
    except:
        return []


# ============================================
# 📊 VERİ ANALİZİ SINIFI
# ============================================

class DataAnalyzer:
    """
    🎯 AMAÇ: TMDB veri setlerini yükler, temizler ve analiz eder
    
    📊 ÖZELLİKLER:
    - Veri yükleme ve birleştirme
    - Eksik veri analizi
    - İstatistiksel özet
    - Görselleştirmeler (tür dağılımı, puan dağılımı, vb.)
    """
    
    def __init__(self, movies_path='tmdb_5000_movies.csv', 
                 credits_path='tmdb_5000_credits.csv',
                 tv_path='TMDB_tv_dataset_v3.csv'):
        """
        🔧 BAŞLATICI: Veri dosya yollarını ayarlar
        """
        self.movies_path = movies_path
        self.credits_path = credits_path
        self.tv_path = tv_path
        
        # Veri çerçeveleri
        self.movies_raw = None
        self.credits_raw = None
        self.tv_raw = None
        self.movies = None
        self.tv = None
        self.combined = None
        
    def load_data(self):
        """
        📥 VERİ YÜKLEME
        🎯 AMAÇ: CSV dosyalarından verileri yükler
        💡 NOT: Dosyalar büyük olduğu için biraz zaman alabilir
        """
        print("📥 Veriler yükleniyor...")
        
        # Film verisi
        self.movies_raw = pd.read_csv(self.movies_path)
        print(f"   ✅ Filmler: {len(self.movies_raw):,} kayıt")
        
        # Oyuncu/Ekip verisi
        self.credits_raw = pd.read_csv(self.credits_path)
        print(f"   ✅ Oyuncular: {len(self.credits_raw):,} kayıt")
        
        # Dizi verisi
        self.tv_raw = pd.read_csv(self.tv_path)
        print(f"   ✅ Diziler: {len(self.tv_raw):,} kayıt")
        
        return self
    
    def preprocess_movies(self):
        """
        🧹 FİLM VERİSİ HAZIRLIĞI
        🎯 AMAÇ: Film ve oyuncu verilerini birleştirir, temizler
        
        📊 İŞLEMLER:
        1. Film ve credits tablolarını birleştir
        2. Türleri parse et
        3. Anahtar kelimeleri parse et
        4. Yönetmen bilgisini çıkar
        5. Oyuncu listesini çıkar
        """
        print("🧹 Film verisi hazırlanıyor...")
        
        # Tabloları birleştir
        self.movies = self.movies_raw.merge(
            self.credits_raw, 
            left_on='id', 
            right_on='movie_id', 
            how='left',
            suffixes=('', '_credits')
        )
        
        # Türleri parse et
        self.movies['genres_list'] = self.movies['genres'].apply(parse_json_column)
        self.movies['genres_str'] = self.movies['genres_list'].apply(lambda x: ', '.join(x))
        
        # Anahtar kelimeleri parse et
        self.movies['keywords_list'] = self.movies['keywords'].apply(parse_json_column)
        self.movies['keywords_str'] = self.movies['keywords_list'].apply(lambda x: ', '.join(x))
        
        # Yönetmen bilgisi
        self.movies['director'] = self.movies['crew'].apply(extract_director)
        
        # Oyuncu listesi
        self.movies['cast_list'] = self.movies['cast'].apply(lambda x: extract_top_cast(x, 5))
        self.movies['cast_str'] = self.movies['cast_list'].apply(lambda x: ', '.join(x))
        
        # Eksik değerleri doldur
        self.movies['overview'] = self.movies['overview'].fillna('')
        self.movies['tagline'] = self.movies['tagline'].fillna('')
        
        # İçerik türü ekle
        self.movies['content_type'] = 'Film'
        
        print(f"   ✅ {len(self.movies):,} film hazırlandı")
        return self
    
    def preprocess_tv(self):
        """
        🧹 DİZİ VERİSİ HAZIRLIĞI
        🎯 AMAÇ: Dizi verisini temizler ve formatlar
        """
        print("🧹 Dizi verisi hazırlanıyor...")
        
        self.tv = self.tv_raw.copy()
        
        # Sütun adlarını düzenle
        self.tv = self.tv.rename(columns={'name': 'title'})
        
        # Türleri parse et (dizi verisinde farklı format olabilir)
        def parse_tv_genres(genre_str):
            if pd.isna(genre_str):
                return []
            return [g.strip() for g in str(genre_str).split(',')]
        
        self.tv['genres_list'] = self.tv['genres'].apply(parse_tv_genres)
        self.tv['genres_str'] = self.tv['genres'].fillna('')
        
        # Eksik değerleri doldur
        self.tv['overview'] = self.tv['overview'].fillna('')
        self.tv['tagline'] = self.tv['tagline'].fillna('')
        
        # İçerik türü ekle
        self.tv['content_type'] = 'Dizi'
        
        # Sadece puan ve oy sayısı olan dizileri al
        self.tv = self.tv[self.tv['vote_count'] > 0]
        
        print(f"   ✅ {len(self.tv):,} dizi hazırlandı")
        return self
    
    def get_stats_summary(self):
        """
        📊 İSTATİSTİKSEL ÖZET
        🎯 AMAÇ: Veri setinin genel istatistiklerini döndürür
        """
        stats = {
            'film_sayisi': len(self.movies) if self.movies is not None else 0,
            'dizi_sayisi': len(self.tv) if self.tv is not None else 0,
            'toplam_icerik': 0,
            'film_ortalama_puan': 0,
            'dizi_ortalama_puan': 0,
            'benzersiz_turler': set(),
            'film_eksik_veri': {},
            'dizi_eksik_veri': {}
        }
        
        if self.movies is not None:
            stats['toplam_icerik'] += len(self.movies)
            stats['film_ortalama_puan'] = self.movies['vote_average'].mean()
            for genre_list in self.movies['genres_list']:
                stats['benzersiz_turler'].update(genre_list)
            stats['film_eksik_veri'] = self.movies.isnull().sum().to_dict()
            
        if self.tv is not None:
            stats['toplam_icerik'] += len(self.tv)
            stats['dizi_ortalama_puan'] = self.tv['vote_average'].mean()
            for genre_list in self.tv['genres_list']:
                stats['benzersiz_turler'].update(genre_list)
            stats['dizi_eksik_veri'] = self.tv.isnull().sum().to_dict()
        
        stats['benzersiz_tur_sayisi'] = len(stats['benzersiz_turler'])
        
        return stats
    
    def get_genre_distribution(self, content_type='Film'):
        """
        📊 TÜR DAĞILIMI
        🎯 AMAÇ: Film/dizi türlerinin dağılımını hesaplar
        📈 ÇIKTI: (tür_adı, sayı) listesi
        """
        if content_type == 'Film' and self.movies is not None:
            all_genres = []
            for genres in self.movies['genres_list']:
                all_genres.extend(genres)
        elif content_type == 'Dizi' and self.tv is not None:
            all_genres = []
            for genres in self.tv['genres_list']:
                all_genres.extend(genres)
        else:
            return []
        
        return Counter(all_genres).most_common(20)
    
    def get_rating_distribution(self, content_type='Film'):
        """
        📊 PUAN DAĞILIMI
        🎯 AMAÇ: Puanların dağılımını döndürür
        """
        if content_type == 'Film' and self.movies is not None:
            return self.movies['vote_average'].dropna()
        elif content_type == 'Dizi' and self.tv is not None:
            return self.tv['vote_average'].dropna()
        return pd.Series([])
    
    def get_popularity_vs_rating(self, content_type='Film'):
        """
        📊 POPÜLERLİK vs PUAN
        🎯 AMAÇ: Popülerlik ve puan arasındaki ilişkiyi döndürür
        """
        if content_type == 'Film' and self.movies is not None:
            return self.movies[['popularity', 'vote_average']].dropna()
        elif content_type == 'Dizi' and self.tv is not None:
            return self.tv[['popularity', 'vote_average']].dropna()
        return pd.DataFrame()
    
    def get_correlation_matrix(self, content_type='Film'):
        """
        📊 KORELASYON MATRİSİ
        🎯 AMAÇ: Sayısal değişkenler arasındaki korelasyonu hesaplar
        💡 AÇIKLAMA: 1'e yakın = güçlü pozitif ilişki, -1'e yakın = güçlü negatif
        """
        if content_type == 'Film' and self.movies is not None:
            numeric_cols = ['budget', 'revenue', 'runtime', 'popularity', 
                          'vote_average', 'vote_count']
            available_cols = [c for c in numeric_cols if c in self.movies.columns]
            return self.movies[available_cols].corr()
        elif content_type == 'Dizi' and self.tv is not None:
            numeric_cols = ['number_of_seasons', 'number_of_episodes', 
                          'popularity', 'vote_average', 'vote_count']
            available_cols = [c for c in numeric_cols if c in self.tv.columns]
            return self.tv[available_cols].corr()
        return pd.DataFrame()
    
    def get_top_content(self, content_type='Film', by='vote_average', n=10):
        """
        🏆 EN İYİ İÇERİKLER
        🎯 AMAÇ: Belirli kritere göre en iyi içerikleri döndürür
        """
        if content_type == 'Film' and self.movies is not None:
            # Minimum oy sayısı filtresi (güvenilirlik için)
            filtered = self.movies[self.movies['vote_count'] >= 100]
            return filtered.nlargest(n, by)[['title', by, 'genres_str', 'vote_count']]
        elif content_type == 'Dizi' and self.tv is not None:
            filtered = self.tv[self.tv['vote_count'] >= 50]
            return filtered.nlargest(n, by)[['title', by, 'genres_str', 'vote_count']]
        return pd.DataFrame()
    
    def get_feature_summary(self):
        """
        📋 ÖZELLİK ÖZETİ
        🎯 AMAÇ: ML için kullanılacak özelliklerin özetini döndürür
        """
        features = {
            'Film Özellikleri': {
                'Metin Özellikleri': ['overview', 'tagline', 'keywords_str'],
                'Kategorik Özellikler': ['genres_str', 'director', 'cast_str', 'original_language'],
                'Sayısal Özellikler': ['budget', 'revenue', 'runtime', 'popularity', 'vote_average', 'vote_count']
            },
            'Dizi Özellikleri': {
                'Metin Özellikleri': ['overview', 'tagline'],
                'Kategorik Özellikler': ['genres_str', 'created_by', 'networks', 'original_language'],
                'Sayısal Özellikler': ['number_of_seasons', 'number_of_episodes', 'popularity', 'vote_average', 'vote_count']
            }
        }
        return features


# ============================================
# 📈 GÖRSEL ANALİZ FONKSİYONLARI
# ============================================

def plot_genre_distribution(genre_counts, title="Tür Dağılımı"):
    """
    📊 TÜR DAĞILIMI GRAFİĞİ
    🎯 AMAÇ: Türlerin yatay bar grafiğini çizer
    """
    if not genre_counts:
        return None
    
    genres = [g[0] for g in genre_counts]
    counts = [g[1] for g in genre_counts]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(genres)))
    bars = ax.barh(genres, counts, color=colors)
    
    ax.set_xlabel('İçerik Sayısı', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Değerleri bar üzerine yaz
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2, 
                f'{count:,}', va='center', fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_rating_distribution(ratings, title="Puan Dağılımı"):
    """
    📊 PUAN DAĞILIMI HİSTOGRAMI
    🎯 AMAÇ: Puanların histogram grafiğini çizer
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(ratings, bins=20, color='steelblue', edgecolor='white', alpha=0.7)
    ax.axvline(ratings.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Ortalama: {ratings.mean():.2f}')
    ax.axvline(ratings.median(), color='orange', linestyle='--', linewidth=2, 
               label=f'Medyan: {ratings.median():.2f}')
    
    ax.set_xlabel('Puan', fontsize=12)
    ax.set_ylabel('İçerik Sayısı', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(corr_matrix, title="Korelasyon Matrisi"):
    """
    📊 KORELASYON ISI HARİTASI
    🎯 AMAÇ: Değişkenler arası ilişkiyi görselleştirir
    💡 AÇIKLAMA: Koyu renkler güçlü ilişkiyi gösterir
    """
    if corr_matrix.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='RdYlBu_r', center=0, ax=ax,
                square=True, linewidths=0.5)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_scatter(data, x_col, y_col, title="Dağılım Grafiği"):
    """
    📊 DAĞILIM GRAFİĞİ (SCATTER PLOT)
    🎯 AMAÇ: İki değişken arasındaki ilişkiyi noktalarla gösterir
    """
    if data.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(data[x_col], data[y_col], alpha=0.5, c='steelblue', s=30)
    ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


# ============================================
# 🧪 TEST KODU
# ============================================

if __name__ == "__main__":
    print("=" * 50)
    print("🎬 TMDB Veri Analizi Modülü")
    print("=" * 50)
    
    # Veri analizörünü başlat
    analyzer = DataAnalyzer()
    
    # Verileri yükle
    analyzer.load_data()
    
    # Verileri hazırla
    analyzer.preprocess_movies()
    analyzer.preprocess_tv()
    
    # İstatistikleri göster
    stats = analyzer.get_stats_summary()
    print("\n📊 VERİ SETİ İSTATİSTİKLERİ")
    print("-" * 30)
    print(f"   Film sayısı: {stats['film_sayisi']:,}")
    print(f"   Dizi sayısı: {stats['dizi_sayisi']:,}")
    print(f"   Toplam içerik: {stats['toplam_icerik']:,}")
    print(f"   Film ortalama puan: {stats['film_ortalama_puan']:.2f}")
    print(f"   Dizi ortalama puan: {stats['dizi_ortalama_puan']:.2f}")
    print(f"   Benzersiz tür sayısı: {stats['benzersiz_tur_sayisi']}")
    
    # En iyi filmler
    print("\n🏆 EN YÜKSEK PUANLI FİLMLER")
    print("-" * 30)
    top_movies = analyzer.get_top_content('Film', 'vote_average', 5)
    print(top_movies.to_string(index=False))
    
    print("\n✅ Veri analizi modülü başarıyla çalışıyor!")
