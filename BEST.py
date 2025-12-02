import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class FinalChampion:
    def __init__(self):
        self.user_stats = {}
        self.book_stats = {}
        self.global_stats = {}
        
    def smart_read_csv(self, filename):
        """Чтение CSV"""
        for sep in [';', ',', '\t', '|']:
            try:
                df = pd.read_csv(filename, sep=sep, encoding='utf-8')
                if len(df.columns) > 1:
                    return df
            except:
                continue
        return pd.read_csv(filename, encoding='latin-1')
    
    def build_champion_features(self):
        """Признаки на основе 0.7715"""
        print("ПРИЗНАКИ...")
        
        train = self.smart_read_csv('train.csv')
        test = self.smart_read_csv('test.csv')
        
        user_col = [c for c in train.columns if 'user' in c.lower() or 'id' in c.lower()][0]
        book_col = [c for c in train.columns if 'book' in c.lower() or 'item' in c.lower()][0]
        rating_col = [c for c in train.columns if 'rating' in c.lower() or 'score' in c.lower()][0]
        
        train = train.rename(columns={user_col: 'user_id', book_col: 'book_id', rating_col: 'rating'})
        test = test.rename(columns={user_col: 'user_id', book_col: 'book_id'})
        
        if 'has_read' in train.columns:
            train = train[train['has_read'] == 1]
        
        # СТАТИСТИКА
        self.global_stats = {
            'mean': train['rating'].mean(),
            'median': train['rating'].median(),
            'std': train['rating'].std(),
            'q1': train['rating'].quantile(0.25),
            'q3': train['rating'].quantile(0.75),
            'mode': train['rating'].mode().iloc[0] if not train['rating'].mode().empty else train['rating'].median()
        }
        
        print(f"   Статистика: mean={self.global_stats['mean']:.3f}, mode={self.global_stats['mode']:.3f}")
        
        # ПРИЗНАКИ - ПОЛЬЗОВАТЕЛИ
        print("Признаки - пользователи...")
        user_stats = train.groupby('user_id').agg({
            'rating': ['mean', 'count', 'std', 'min', 'max', 'median']
        }).reset_index()
        user_stats.columns = ['user_id', 'mean', 'count', 'std', 'min', 'max', 'median']
        
        # ПРОВЕРЕННЫЕ ПРИЗНАКИ С 0.7715
        user_stats['confidence'] = np.log1p(user_stats['count']) / 3.8
        user_stats['generosity'] = (user_stats['mean'] - self.global_stats['mean']) / 1.9
        user_stats['consistency'] = 1 / (1 + user_stats['std'].fillna(1.1))
        user_stats['positivity'] = (user_stats['mean'] > 6.5).astype(float) * 0.2
        
        self.user_stats = {}
        for _, row in user_stats.iterrows():
            self.user_stats[row['user_id']] = {
                'mean': row['mean'],
                'count': row['count'], 
                'confidence': row['confidence'],
                'generosity': row['generosity'],
                'consistency': row['consistency'],
                'min': row['min'],
                'max': row['max'],
                'median': row['median'],
                'positivity': row['positivity']
            }
        
        # ПРИЗНАКИ - КНИГИ
        print("Признаки - книги...")
        book_stats = train.groupby('book_id').agg({
            'rating': ['mean', 'count', 'std', 'min', 'max', 'median']
        }).reset_index()
        book_stats.columns = ['book_id', 'mean', 'count', 'std', 'min', 'max', 'median']
        
        book_stats['popularity'] = np.log1p(book_stats['count']) / 3.5
        book_stats['controversial'] = (book_stats['std'] > 2.1).astype(float)
        book_stats['consistency'] = 1 / (1 + book_stats['std'].fillna(1.1))
        book_stats['high_quality'] = (book_stats['mean'] > 7.2).astype(float) * 0.25
        
        self.book_stats = {}
        for _, row in book_stats.iterrows():
            self.book_stats[row['book_id']] = {
                'mean': row['mean'],
                'count': row['count'],
                'popularity': row['popularity'],
                'controversial': row['controversial'],
                'consistency': row['consistency'],
                'min': row['min'],
                'max': row['max'],
                'median': row['median'],
                'high_quality': row['high_quality']
            }
        
        print(f"   Пользователей: {len(self.user_stats)}, Книг: {len(self.book_stats)}")
        return train, test
    
    def calculate_champion_prediction(self, user_id, book_id):
        """Логика - точная реплика 0.7715 с микрооптимизациями"""
        user = self.user_stats.get(user_id, {})
        book = self.book_stats.get(book_id, {})
        
        # Базовые значения
        user_mean = user.get('mean', self.global_stats['mean'])
        book_mean = book.get('mean', self.global_stats['mean'])
        user_median = user.get('median', self.global_stats['median'])
        book_median = book.get('median', self.global_stats['median'])
        global_mean = self.global_stats['mean']
        global_median = self.global_stats['median']
        
        # ВЗВЕШИВАНИЕ - ТОЧНАЯ ФОРМУЛА 0.7715
        user_conf = user.get('confidence', 0.18)
        book_conf = book.get('popularity', 0.18)
        user_consistency = user.get('consistency', 0.6)
        book_consistency = book.get('consistency', 0.6)
        
        total_conf = user_conf * user_consistency + book_conf * book_consistency + 0.35
        user_weight = (user_conf * user_consistency) / total_conf
        book_weight = (book_conf * book_consistency) / total_conf
        global_weight = 0.35 / total_conf
        
        # КОМБИНИРОВАНИЕ - ФОРМУЛА 0.7715
        user_combined = 0.65 * user_mean + 0.35 * user_median
        book_combined = 0.65 * book_mean + 0.35 * book_median
        global_combined = 0.55 * global_mean + 0.45 * global_median
        
        pred_weighted = (user_combined * user_weight + 
                        book_combined * book_weight + 
                        global_combined * global_weight)
        
        # КОРРЕКТИРОВКИ 
        user_generosity = user.get('generosity', 0)
        generosity_boost = user_generosity * 0.32
        
        user_positivity = user.get('positivity', 0)
        positivity_boost = user_positivity * 0.15
        
        book_controversial = book.get('controversial', 0)
        if book_controversial:
            controversy_adjust = -0.12 * (pred_weighted - global_median)
        else:
            controversy_adjust = 0
        
        book_high_quality = book.get('high_quality', 0)
        quality_boost = book_high_quality * 0.18
        
        # ФИНАЛЬНАЯ КОМБ - ФОРМУЛА 0.7715
        final_pred = (pred_weighted + 
                     generosity_boost + 
                     positivity_boost +
                     controversy_adjust + 
                     quality_boost)
        
        # ОГРАНИЧЕНИЯ ДИАПАЗОНА
        user_min = user.get('min', max(1, global_mean - 2.2))
        user_max = user.get('max', min(10, global_mean + 2.2))
        
        if final_pred < user_min:
            blend_ratio = 0.75 if user.get('count', 0) > 2 else 0.85
            final_pred = blend_ratio * final_pred + (1 - blend_ratio) * user_min
        elif final_pred > user_max:
            blend_ratio = 0.75 if user.get('count', 0) > 2 else 0.85
            final_pred = blend_ratio * final_pred + (1 - blend_ratio) * user_max
        
        # ОБРАБОТКА НОВЫХ ДАННЫХ
        user_count = user.get('count', 0)
        book_count = book.get('count', 0)
        
        if user_count < 2 or book_count < 2:
            newness_penalty = max(0, 0.55 - 0.1 * min(user_count, book_count))
            final_pred = (1 - newness_penalty) * final_pred + newness_penalty * global_combined
        
        # БУСТ - ОПТИМАЛЬНЫЙ
        if user_count >= 3 and book_count >= 3:
            final_pred = final_pred * 1.019  # ИДЕАЛЬНЫЙ БУСТ - между 1.018 и 1.022
        elif user_count >= 1 or book_count >= 1:
            final_pred = final_pred * 1.009  # Оптимальный средний
        else:
            final_pred = final_pred * 1.003  # Консервативный для новых
        
        return final_pred
    
    def apply_champion_calibration(self, predictions, train_ratings):
        """Идеальная калибровка"""
        pred_array = np.array(predictions)
        train_array = train_ratings.values
        
        # СОХРАНЕНИЕ РАСПРЕДЕЛЕНИЯ
        target_mean = np.mean(train_array)
        target_median = np.median(train_array)
        target_std = np.std(train_array)
        
        current_mean = np.mean(pred_array)
        current_median = np.median(pred_array)
        current_std = np.std(pred_array)
        
        # ТОЧНАЯ КОРРЕКТИРОВКА СРЕДНЕГО

        mean_diff = target_mean - current_mean
        if abs(mean_diff) > 0.07:
            adjustment = mean_diff * 0.22
            pred_array = pred_array + adjustment
        
        # ТОЧНАЯ КОРРЕКТИРОВКА МЕДИАНЫ

        median_diff = target_median - current_median
        if abs(median_diff) > 0.08:
            pred_array = pred_array + median_diff * 0.14
        
        # ИДЕАЛЬНОЕ СОХРАНЕНИЕ СТАНДАРТНОГО ОТКЛОНЕНИЯ

        if current_std > 0 and target_std > 0:
            std_ratio = target_std / current_std
            if 0.92 < std_ratio < 1.08:
                centered = pred_array - np.mean(pred_array)
                pred_array = centered * (std_ratio ** 0.92) + np.mean(pred_array)
        
        # МИКРОКОРРЕКТИРОВКА КВАНТИЛЕЙ

        quantiles = [0.1, 0.25, 0.75, 0.9]
        for q in quantiles:
            current_q = np.quantile(pred_array, q)
            target_q = np.quantile(train_array, q)
            diff = target_q - current_q
            
            if abs(diff) > 0.12:
                if q > 0.5:
                    mask = pred_array >= current_q
                else:
                    mask = pred_array <= current_q
                
                weight = 0.06 if q in [0.1, 0.9] else 0.04
                pred_array[mask] = pred_array[mask] + diff * weight
        
        return np.clip(pred_array, 1.0, 10.0)
    
    def run_champion(self):
        """ЗАПУСК"""
        print("ЗАПУСК!")
        
        try:
            train, test = self.build_champion_features()
            
            print("Создание предсказаний...")
            predictions = []
            for i, row in test.iterrows():
                pred = self.calculate_champion_prediction(row['user_id'], row['book_id'])
                predictions.append(pred)
            
            print("Применение идеальной калибровки...")
            calibrated_predictions = self.apply_champion_calibration(predictions, train['rating'])
            
            submission = test[['user_id', 'book_id']].copy()
            submission['rating_predict'] = calibrated_predictions
            
            self.champion_analysis(submission, train)
            
            submission.to_csv('final_champion.csv', index=False)
            
            print(f"\n ФАЙЛ final_champion.csv СОЗДАН!")
            
            return submission
            
        except Exception as e:
            print(f"Критическая ошибка: {e}")
            return self.create_perfect_backup()
    
    def create_perfect_backup(self):
        """Идеальное резервное решение"""
        print("🛡️ СОЗДАНИЕ ИДЕАЛЬНОГО РЕЗЕРВА...")
        
        train = self.smart_read_csv('train.csv')
        test = self.smart_read_csv('test.csv')
        
        user_col = train.columns[0]
        book_col = train.columns[1]
        
        train = train.rename(columns={user_col: 'user_id', book_col: 'book_id'})
        test = test.rename(columns={user_col: 'user_id', book_col: 'book_id'})
        
        rating_col = [c for c in train.columns if 'rating' in c.lower()][0]
        train = train.rename(columns={rating_col: 'rating'})
        
        if 'has_read' in train.columns:
            train = train[train['has_read'] == 1]
        
        user_means = train.groupby('user_id')['rating'].mean()
        book_means = train.groupby('book_id')['rating'].mean()
        global_mean = train['rating'].mean()
        
        predictions = []
        for _, row in test.iterrows():
            user_pred = user_means.get(row['user_id'], global_mean)
            book_pred = book_means.get(row['book_id'], global_mean)
            pred = (user_pred * 0.65 + book_pred * 0.35) * 1.018  # Формула
            predictions.append(pred)
        
        submission = test[['user_id', 'book_id']].copy()
        submission['rating_predict'] = np.clip(predictions, 1, 10)
        submission.to_csv('perfect_backup.csv', index=False)
        
        print("Cоздан!")
        return submission
    
    def champion_analysis(self, submission, train):
        """Анализ"""
        print("\nАНАЛИЗ:")
        
        pred_stats = submission['rating_predict'].describe()
        train_stats = train['rating'].describe()
        
        print(f"Предсказания: {pred_stats['min']:.3f} - {pred_stats['max']:.3f}")
        print(f"Среднее: {pred_stats['mean']:.3f} (трейнинг: {train_stats['mean']:.3f})")
        print(f"Медиана: {np.median(submission['rating_predict']):.3f} (трейнинг: {train_stats['50%']:.3f})")
        print(f"Стандартное отклонение: {pred_stats['std']:.3f} (трейнинг: {train_stats['std']:.3f})")
        
        mean_diff = abs(pred_stats['mean'] - train_stats['mean'])
        median_diff = abs(np.median(submission['rating_predict']) - train_stats['50%'])
        
        print(f"\n Идеальная калибровка:")
        print(f"   Среднее: {'✓' if mean_diff < 0.04 else 'err'} (разница: {mean_diff:.3f})")
        print(f"   Медиана: {'✓' if median_diff < 0.05 else 'err'} (разница: {median_diff:.3f})")

# ЗАПУСК
if __name__ == "__main__":
    print("ДЕЛАЕМ ИСТОРИЮ!")
    print("=" * 70)
    
    champion = FinalChampion()
    submission = champion.run_champion()
    
    print(f"\n РЕШЕНИЕ ГОТОВО!")