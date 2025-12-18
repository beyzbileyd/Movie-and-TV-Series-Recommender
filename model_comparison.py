# ============================================
# 📈 MODEL KARŞILAŞTIRMA MODÜLÜ
# ============================================
# 🎯 AMAÇ: Farklı ML modellerini karşılaştırır ve en iyisini seçer
# 📝 AÇIKLAMA: Bu modül modellerin performansını çeşitli metriklerle
#              değerlendirir ve görselleştirir.
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import time
import warnings
warnings.filterwarnings('ignore')


# ============================================
# 📊 KARŞILAŞTIRMA SINIFI
# ============================================

class ModelComparator:
    """
    🎯 AMAÇ: Birden fazla ML modelini karşılaştırır
    
    📊 ÖZELLİKLER:
    - Eğitim süresi karşılaştırması
    - Öneri kalitesi değerlendirmesi
    - Çeşitlilik ve kapsam analizi
    - Görsel karşılaştırma grafikleri
    - En iyi model seçimi
    """
    
    def __init__(self, models):
        """
        🔧 BAŞLATICI
        📊 PARAMETRE:
        - models: Model sözlüğü {'model_adı': model_instance}
        """
        self.models = models
        self.results = {}
        self.comparison_df = None
        self.best_model_name = None
        
    def evaluate_all(self, df, test_items=None, n_recommendations=10):
        """
        🎓 TÜM MODELLERİ DEĞERLENDİR
        
        📊 ADIMLAR:
        1. Her modeli eğit
        2. Test filmler için öneri al
        3. Metrikleri hesapla
        4. Sonuçları karşılaştır
        """
        print("=" * 60)
        print("📈 Model Karşılaştırma Başlıyor")
        print("=" * 60)
        
        # Test filmleri seç
        if test_items is None:
            # Popüler filmlerden rastgele seç
            popular = df[df['vote_count'] >= 100].sample(min(20, len(df)))
            test_items = popular['title'].tolist()
        
        for model_name, model in self.models.items():
            print(f"\n🔄 Değerlendiriliyor: {model.name}")
            
            result = {
                'model_name': model.name,
                'fit_time': 0,
                'avg_recommendation_time': 0,
                'coverage': 0,
                'diversity': 0,
                'avg_rating': 0,
                'recommendations': []
            }
            
            # Model eğitimi
            try:
                start_time = time.time()
                model.fit(df)
                result['fit_time'] = time.time() - start_time
                print(f"   ⏱️ Eğitim süresi: {result['fit_time']:.2f}s")
            except Exception as e:
                print(f"   ❌ Eğitim hatası: {e}")
                continue
            
            # Öneri alma
            all_recommendations = set()
            all_genres = []
            all_ratings = []
            rec_times = []
            
            for title in test_items:
                try:
                    start_time = time.time()
                    recs = model.recommend(title, n=n_recommendations)
                    rec_times.append(time.time() - start_time)
                    
                    if not recs.empty:
                        all_recommendations.update(recs['title'].tolist())
                        
                        if 'genres_str' in recs.columns:
                            all_genres.extend(recs['genres_str'].tolist())
                        
                        if 'vote_average' in recs.columns:
                            all_ratings.extend(recs['vote_average'].tolist())
                        
                        result['recommendations'].append({
                            'query': title,
                            'results': recs.to_dict('records')
                        })
                except Exception as e:
                    print(f"   ⚠️ Öneri hatası ({title}): {e}")
            
            # Metrikleri hesapla
            result['avg_recommendation_time'] = np.mean(rec_times) if rec_times else 0
            result['coverage'] = len(all_recommendations) / len(df) * 100  # Yüzde
            result['diversity'] = len(set(all_genres)) / max(len(all_genres), 1) * 100  # Yüzde
            result['avg_rating'] = np.mean(all_ratings) if all_ratings else 0
            
            print(f"   📊 Kapsam: {result['coverage']:.2f}%")
            print(f"   🎭 Çeşitlilik: {result['diversity']:.2f}%")
            print(f"   ⭐ Ort. Puan: {result['avg_rating']:.2f}")
            
            self.results[model_name] = result
        
        # Karşılaştırma DataFrame'i oluştur
        self._create_comparison_df()
        
        # En iyi modeli seç
        self._select_best_model()
        
        return self
    
    def _create_comparison_df(self):
        """
        📊 KARŞILAŞTIRMA TABLOSU OLUŞTUR
        """
        data = []
        for model_name, result in self.results.items():
            data.append({
                'Model': result['model_name'],
                'Eğitim Süresi (s)': result['fit_time'],
                'Öneri Süresi (s)': result['avg_recommendation_time'],
                'Kapsam (%)': result['coverage'],
                'Çeşitlilik (%)': result['diversity'],
                'Ort. Puan': result['avg_rating']
            })
        
        self.comparison_df = pd.DataFrame(data)
        
    def _select_best_model(self):
        """
        🏆 EN İYİ MODELİ SEÇ
        
        📊 SKOR HESAPLAMA:
        - Kapsam: %30 ağırlık
        - Çeşitlilik: %30 ağırlık
        - Ort. Puan: %20 ağırlık
        - Hız: %20 ağırlık (düşük = iyi)
        """
        if self.comparison_df is None or self.comparison_df.empty:
            return
        
        df = self.comparison_df.copy()
        
        # Normalize et (0-1 aralığına)
        def normalize(series):
            min_val, max_val = series.min(), series.max()
            if max_val == min_val:
                return pd.Series([0.5] * len(series))
            return (series - min_val) / (max_val - min_val)
        
        # Süre için ters normalize (düşük = iyi)
        def normalize_inverse(series):
            return 1 - normalize(series)
        
        # Skorları hesapla
        df['skor_kapsam'] = normalize(df['Kapsam (%)']) * 0.30
        df['skor_cesitlilik'] = normalize(df['Çeşitlilik (%)']) * 0.30
        df['skor_puan'] = normalize(df['Ort. Puan']) * 0.20
        df['skor_hiz'] = normalize_inverse(df['Eğitim Süresi (s)'] + df['Öneri Süresi (s)']) * 0.20
        
        df['Toplam Skor'] = df['skor_kapsam'] + df['skor_cesitlilik'] + df['skor_puan'] + df['skor_hiz']
        
        # En iyi modeli bul
        best_idx = df['Toplam Skor'].idxmax()
        self.best_model_name = df.loc[best_idx, 'Model']
        
        # Skor sütununu ana DataFrame'e ekle
        self.comparison_df['Toplam Skor'] = df['Toplam Skor']
        self.comparison_df = self.comparison_df.sort_values('Toplam Skor', ascending=False)
        
        print(f"\n🏆 EN İYİ MODEL: {self.best_model_name}")
        
    def get_comparison_table(self):
        """
        📊 KARŞILAŞTIRMA TABLOSUNU DÖNDÜR
        """
        return self.comparison_df
    
    def get_best_model(self):
        """
        🏆 EN İYİ MODELİ DÖNDÜR
        """
        for model_name, model in self.models.items():
            if model.name == self.best_model_name:
                return model
        return None
    
    def get_detailed_metrics(self):
        """
        📋 DETAYLI METRİKLERİ DÖNDÜR
        """
        detailed = []
        for model_name, result in self.results.items():
            detailed.append({
                'Model': result['model_name'],
                'Eğitim Süresi': f"{result['fit_time']:.3f}s",
                'Öneri Süresi': f"{result['avg_recommendation_time']:.4f}s",
                'Kapsam': f"{result['coverage']:.2f}%",
                'Çeşitlilik': f"{result['diversity']:.2f}%",
                'Ort. Puan': f"{result['avg_rating']:.2f}",
                'Önerilen Film Sayısı': len(result['recommendations'])
            })
        return pd.DataFrame(detailed)


# ============================================
# 📈 GÖRSELLEŞTIRME FONKSİYONLARI
# ============================================

def plot_training_time_comparison(comparison_df):
    """
    📊 EĞİTİM SÜRESİ KARŞILAŞTIRMASI
    🎯 AMAÇ: Modellerin eğitim sürelerini karşılaştırır
    """
    if comparison_df is None or comparison_df.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(comparison_df)))
    bars = ax.bar(comparison_df['Model'], comparison_df['Eğitim Süresi (s)'], color=colors)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Eğitim Süresi (saniye)', fontsize=12)
    ax.set_title('📊 Model Eğitim Süresi Karşılaştırması', fontsize=14, fontweight='bold')
    
    # Değerleri bar üzerine yaz
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_coverage_diversity(comparison_df):
    """
    📊 KAPSAM VE ÇEŞİTLİLİK KARŞILAŞTIRMASI
    🎯 AMAÇ: Modellerin kapsam ve çeşitlilik metriklerini gösterir
    """
    if comparison_df is None or comparison_df.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(comparison_df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, comparison_df['Kapsam (%)'], width, 
                   label='Kapsam (%)', color='steelblue')
    bars2 = ax.bar(x + width/2, comparison_df['Çeşitlilik (%)'], width, 
                   label='Çeşitlilik (%)', color='coral')
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Yüzde (%)', fontsize=12)
    ax.set_title('📊 Kapsam ve Çeşitlilik Karşılaştırması', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    return fig

def plot_overall_scores(comparison_df):
    """
    📊 GENEL SKOR KARŞILAŞTIRMASI
    🎯 AMAÇ: Modellerin toplam skorlarını gösterir
    💡 EN YÜKSEK SKORLU MODEL = EN İYİ MODEL
    """
    if comparison_df is None or 'Toplam Skor' not in comparison_df.columns:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sırala
    sorted_df = comparison_df.sort_values('Toplam Skor', ascending=True)
    
    colors = ['gold' if x == sorted_df['Toplam Skor'].max() else 'steelblue' 
              for x in sorted_df['Toplam Skor']]
    
    bars = ax.barh(sorted_df['Model'], sorted_df['Toplam Skor'], color=colors)
    
    ax.set_xlabel('Toplam Skor', fontsize=12)
    ax.set_title('🏆 Model Performans Sıralaması', fontsize=14, fontweight='bold')
    
    # Değerleri bar üzerine yaz
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_radar_comparison(comparison_df):
    """
    📊 RADAR GRAFİĞİ
    🎯 AMAÇ: Modelleri çok boyutlu olarak karşılaştırır
    """
    if comparison_df is None or comparison_df.empty:
        return None
    
    categories = ['Kapsam', 'Çeşitlilik', 'Puan', 'Hız']
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Kapatmak için
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(comparison_df)))
    
    for idx, (_, row) in enumerate(comparison_df.iterrows()):
        # Metrikleri normalize et (0-1)
        values = [
            row['Kapsam (%)'] / 100,
            row['Çeşitlilik (%)'] / 100,
            row['Ort. Puan'] / 10,
            1 - min(row['Eğitim Süresi (s)'] / 10, 1)  # Hız (ters)
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_title('📊 Model Karşılaştırma Radar Grafiği', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    return fig


# ============================================
# 🏆 EN İYİ MODEL SEÇİCİ
# ============================================

def select_best_model_for_task(comparison_df, priority='balanced'):
    """
    🏆 GÖREVE GÖRE EN İYİ MODEL SEÇ
    
    📊 ÖNCELİKLER:
    - 'balanced': Dengeli (varsayılan)
    - 'speed': Hız öncelikli
    - 'quality': Kalite öncelikli
    - 'coverage': Kapsam öncelikli
    """
    if comparison_df is None or comparison_df.empty:
        return None
    
    if priority == 'speed':
        # En hızlı model
        return comparison_df.loc[comparison_df['Eğitim Süresi (s)'].idxmin(), 'Model']
    
    elif priority == 'quality':
        # En yüksek puanlı öneriler yapan model
        return comparison_df.loc[comparison_df['Ort. Puan'].idxmax(), 'Model']
    
    elif priority == 'coverage':
        # En geniş kapsamlı model
        return comparison_df.loc[comparison_df['Kapsam (%)'].idxmax(), 'Model']
    
    else:  # balanced
        # Toplam skor en yüksek
        if 'Toplam Skor' in comparison_df.columns:
            return comparison_df.loc[comparison_df['Toplam Skor'].idxmax(), 'Model']
        return comparison_df.iloc[0]['Model']


# ============================================
# 🧪 TEST KODU
# ============================================

if __name__ == "__main__":
    from data_analysis import DataAnalyzer
    from ml_models import get_all_models
    
    print("=" * 60)
    print("📈 Model Karşılaştırma Testi")
    print("=" * 60)
    
    # Veri yükle
    analyzer = DataAnalyzer()
    analyzer.load_data()
    analyzer.preprocess_movies()
    
    # Sadece test amaçlı küçük veri seti
    test_df = analyzer.movies.head(500)
    
    # Modelleri al
    models = get_all_models()
    
    # Karşılaştırıcı oluştur
    comparator = ModelComparator(models)
    
    # Değerlendir
    comparator.evaluate_all(test_df, n_recommendations=5)
    
    # Sonuçları göster
    print("\n" + "=" * 60)
    print("📊 KARŞILAŞTIRMA SONUÇLARI")
    print("=" * 60)
    print(comparator.get_comparison_table().to_string(index=False))
    
    print(f"\n🏆 Seçilen En İyi Model: {comparator.best_model_name}")
    
    print("\n✅ Model karşılaştırma modülü başarıyla çalışıyor!")
