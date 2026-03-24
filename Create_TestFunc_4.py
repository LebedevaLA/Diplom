import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from GenerateFunc_1 import UnimodalGenerator, PeriodicGenerator, PiecewiseGenerator, plot_and_save_functions


# =====СЛОЖНЫЕ УНИМОДАЛЬНЫЕ ФУНКЦИИ =====

class ComplexUnimodal:
    """
    Класс сложных унимодальных функций.
    Комбинации базовых функций, сохраняющие один глобальный минимум/максимум.
    """
    
    def __init__(self, func_type=None):
        self.types = [
            'quadratic_log',      # парабола + логарифм
            'quadratic_exp',      # парабола + экспонента
            'gaussian_poly',      # гауссиан + полином
            'double_quadratic',   # две параболы с разными весами
        ]
        
        if func_type is None:
            self.func_type = np.random.choice(self.types)
        else:
            self.func_type = func_type
            
        self.params_dict = self._generate_random_params()
        self.param_names = list(self.params_dict.keys())
        self.true_params = list(self.params_dict.values())
    
    def _generate_random_params(self):
        params = {}
        
        if self.func_type == 'quadratic_log':
            # a*(x - b)² + c + d*log((x - e)² + 1)
            params['a'] = np.random.uniform(0.5, 5.0)      # вес параболы
            params['b'] = np.random.uniform(-8, 8)        # центр параболы
            params['c'] = np.random.uniform(-5, 5)        # сдвиг
            params['d'] = np.random.uniform(0.1, 2.0)     # вес логарифма (маленький)
            params['e'] = np.random.uniform(-5, 5)        # центр логарифма
            
        elif self.func_type == 'quadratic_exp':
            # a*(x - b)² + c + d*exp(-(x - e)²) 
            params['a'] = np.random.uniform(0.5, 5.0)
            params['b'] = np.random.uniform(-8, 8)
            params['c'] = np.random.uniform(-5, 5)
            params['d'] = np.random.uniform(0.1, 1.5)     # маленький вес экспоненты
            params['e'] = np.random.uniform(-5, 5)
            
        elif self.func_type == 'gaussian_poly':
            # -a*exp(-(x - mu)²/(2*sigma²)) + c + d*x²
            params['a'] = np.random.uniform(0.5, 5.0)
            params['mu'] = np.random.uniform(-8, 8)
            params['sigma'] = np.random.uniform(0.5, 3.0)
            params['c'] = np.random.uniform(-5, 5)
            params['d'] = np.random.uniform(0.1, 1.0)     # вес полинома
            
            
        return params
    
    def evaluate(self, params_list, x):
        """Вычисляет значение функции в точках x"""
        x = np.asarray(x).flatten()
        
        if self.func_type == 'quadratic_log':
            a, b, c, d, e = params_list
            return a*(x - b)**2 + c + d*np.log((x - e)**2 + 1)
            
        elif self.func_type == 'quadratic_exp':
            a, b, c, d, e = params_list
            return a*(x - b)**2 + c + d*np.exp(-(x - e)**2)
            
        elif self.func_type == 'gaussian_poly':
            a, mu, sigma, c, d = params_list
            return -a*np.exp(-(x - mu)**2/(2*sigma**2)) + c + d*x**2
            
    
    def __call__(self, x):
        return self.evaluate(self.true_params, x)
    
    def get_true_params(self):
        return self.true_params.copy()


# ====СЛОЖНЫЕ ПЕРИОДИЧЕСКИЕ ФУНКЦИИ ====

class ComplexPeriodic:
    """
    Класс сложных периодических функций.
    Комбинации синусов/косинусов с разными частотами.
    """
    
    def __init__(self, func_type=None):
        self.types = [
            'sin_cos_mix',        # sin + cos с разными частотами
            'sin_sin',             # два синуса с разными частотами
            'cos_cos',             # два косинуса с разными частотами
        ]
        
        if func_type is None:
            self.func_type = np.random.choice(self.types)
        else:
            self.func_type = func_type
            
        self.params_dict = self._generate_random_params()
        self.param_names = list(self.params_dict.keys())
        self.true_params = list(self.params_dict.values())
    
    def _generate_random_params(self):
        params = {}
        
        if self.func_type == 'sin_cos_mix':
            params['amp1'] = np.random.uniform(0.5, 4.0)
            params['freq1'] = np.random.uniform(0.5, 3.0)
            params['phase1'] = np.random.uniform(0, 2*np.pi)
            params['amp2'] = np.random.uniform(0.5, 4.0)
            params['freq2'] = np.random.uniform(0.5, 3.0)
            params['phase2'] = np.random.uniform(0, 2*np.pi)
            params['shift'] = np.random.uniform(-3.0, 3.0)
            
        elif self.func_type == 'sin_sin':
            params['amp1'] = np.random.uniform(0.5, 4.0)
            params['freq1'] = np.random.uniform(0.5, 3.0)
            params['phase1'] = np.random.uniform(0, 2*np.pi)
            params['amp2'] = np.random.uniform(0.5, 4.0)
            params['freq2'] = np.random.uniform(0.5, 3.0)
            params['phase2'] = np.random.uniform(0, 2*np.pi)
            params['shift'] = np.random.uniform(-3.0, 3.0)
            
        elif self.func_type == 'cos_cos':
            params['amp1'] = np.random.uniform(0.5, 4.0)
            params['freq1'] = np.random.uniform(0.1, 0.5)
            params['phase1'] = np.random.uniform(0, 2*np.pi)
            params['amp2'] = np.random.uniform(0.5, 4.0)
            params['freq2'] = np.random.uniform(0.1, 0.5)
            params['phase2'] = np.random.uniform(0, 2*np.pi)
            params['shift'] = np.random.uniform(-3.0, 3.0)
            
            
        return params
    
    def evaluate(self, params_list, x):
        x = np.asarray(x).flatten()
        
        if self.func_type == 'sin_cos_mix':
            amp1, freq1, phase1, amp2, freq2, phase2, shift = params_list
            return amp1*np.sin(freq1*x + phase1) + amp2*np.cos(freq2*x + phase2) + shift
            
        elif self.func_type == 'sin_sin':
            amp1, freq1, phase1, amp2, freq2, phase2, shift = params_list
            return amp1*np.sin(freq1*x + phase1) + amp2*np.sin(freq2*x + phase2) + shift
            
        elif self.func_type == 'cos_cos':
            amp1, freq1, phase1, amp2, freq2, phase2, shift = params_list
            return amp1*np.cos(freq1*x + phase1) + amp2*np.cos(freq2*x + phase2) + shift
            
    def __call__(self, x):
        return self.evaluate(self.true_params, x)
    
    def get_true_params(self):
        return self.true_params.copy()


# ==== ГЕНЕРАТОРЫ ====

class ComplexUnimodalGenerator:
    def __init__(self, x_vals):
        self.x_vals = x_vals
    
    def generate(self, n_functions=25, st = 25):
        functions = []
        for i in range(n_functions):
            func = ComplexUnimodal()
            y = func(self.x_vals)
            functions.append({
                'class': 'complex_unimodal',
                'type': func.func_type,
                'true_params': func.get_true_params(),
                'func_obj': func,
                'x': self.x_vals.copy(),
                'y': y,
                'id': st+i
            })
        return functions


class ComplexPeriodicGenerator:
    def __init__(self, x_vals):
        self.x_vals = x_vals
    
    def generate(self, n_functions=25,st = 25):
        functions = []
        for i in range(n_functions):
            func = ComplexPeriodic()
            y = func(self.x_vals)
            functions.append({
                'class': 'complex_periodic',
                'type': func.func_type,
                'true_params': func.get_true_params(),
                'func_obj': func,
                'x': self.x_vals.copy(),
                'y': y,
                'id': st+i
            })
        return functions


# Main
if __name__ == "__main__":
    # Загружаем x_vals
    if os.path.exists('x_vals.npy'):
        x_vals = np.load('x_vals.npy')
    else:
        exit(1)

    # Создаем генераторы сложных функций
    print("\n" + "="*60)
    print("ГЕНЕРАЦИЯ СЛОЖНЫХ ФУНКЦИЙ")
    print("="*60)
    
    # Тестовые данные простые функции из старых генерторов + ноемного новых более сложных функций
    simple_unimodal_gen = UnimodalGenerator(x_vals)
    simple_periodic_gen = PeriodicGenerator(x_vals)
    simple_piecewise_gen = PiecewiseGenerator(x_vals)
    complex_unimodal_gen = ComplexUnimodalGenerator(x_vals)
    complex_periodic_gen = ComplexPeriodicGenerator(x_vals)
    

    # Генерируем
    simple_unimodal_data = simple_unimodal_gen.generate(25) 
    simple_periodic_data = simple_periodic_gen.generate(25)
    simple_piecewise_data = simple_piecewise_gen.generate(25)
    complex_unimodal_data = complex_unimodal_gen.generate(5) 
    complex_periodic_data = complex_periodic_gen.generate(5)
    
    # Сохраняем сгенерированные данные
    test_data = {
        'test_unimodal': simple_unimodal_data + complex_unimodal_data,
        'test_periodic': simple_periodic_data + complex_periodic_data,
        'test_piecewise': simple_piecewise_data
    }
    
    with open('test_functions.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    print("\n Данные сохранены в test_functions.pkl")
    
    # Создаем папку для графиков если её нет
    os.makedirs('test_plots', exist_ok=True)
    
    # Рисуем и сохраняем графики
    plot_and_save_functions(
        simple_unimodal_data + complex_unimodal_data, 
        'Test Unimodal', 
        'test_plots/test_unimodal.png',
    )
    
    plot_and_save_functions(
        simple_periodic_data + complex_periodic_data, 
        'Test Periodic', 
        'test_plots/test_periodic.png',
    )
    
    plot_and_save_functions(
        simple_piecewise_data,
        'Test Piecewise', 
        'test_plots/test_piecewise.png',
    )
    print(f"Графики сохранены в папке 'test_plots/'")
    print(f"Данные сохранены в 'test_functions.pkl'")