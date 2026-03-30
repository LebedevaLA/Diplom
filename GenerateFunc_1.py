import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

'===== КЛАСС УНИМОДАЛЬНЫХ 1D ФУНКЦИЙ ====='
class Unimodal:
    #Выбираем тип функции и вызываем функцию генерации параметров
    def __init__(self, function_type=None):
        self.types = ['quadratic', 'absolute', 'gaussian']
        if function_type is None:
            self.func_type = np.random.choice(self.types)
        else:
            self.func_type = function_type

        self.true_params = self._generate_random_params()
    
    #Генерация параметров
    def _generate_random_params(self):
        if self.func_type == 'quadratic':
            a = np.random.uniform(0.1, 10.0) * np.random.choice([-1, 1])
            b = np.random.uniform(-10.0, 10.0)
            c = np.random.uniform(-10.0, 10.0)
            params = [a, b, c]
        elif self.func_type == 'absolute':
            a = np.random.uniform(0.1, 10.0) * np.random.choice([-1, 1])
            b = np.random.uniform(-10.0, 10.0)
            c = np.random.uniform(-10.0, 10.0)
            params = [a, b, c]
        elif self.func_type == 'gaussian':
            a = np.random.uniform(0.1, 10.0) * np.random.choice([-1, 1])
            mu = np.random.uniform(-10.0, 10.0)
            sigma = np.random.uniform(0.5, 5.0)
            c = np.random.uniform(-10.0, 10.0)
            params = [a, mu, sigma, c]
        
        return params
    
    #Считаем функцию с заданнами параметрами и x
    def evaluate(self, params_list, x):
        x = np.asarray(x).flatten()
        if self.func_type == 'quadratic':
            a, b, c = params_list
            return a * (x - b)**2 + c
        elif self.func_type == 'absolute':
            a, b, c = params_list
            return a * np.abs(x - b) + c
        elif self.func_type == 'gaussian':
            a, mu, sigma, c = params_list
            return -a * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c

    
    #специальный метод в Python - вызывать объект как функцию используется только в генераторе для удабства
    def __call__(self, x):
        return self.evaluate(self.true_params, x)

    def get_true_params(self):
        return self.true_params.copy()


'===== КЛАСС ПЕРИОДИЧЕСКИХ 1D ФУНКЦИЙ ====='
class Periodic:
    def __init__(self, func_type=None):
        self.types = ['sin', 'cos', 'sin_abs', 'sin_squared', 'sin_cos']
        if func_type is None:
            self.func_type = np.random.choice(self.types)
        else:
            self.func_type = func_type

        self.true_params = self._generate_random_params()

    def _generate_random_params(self):
        if self.func_type in ['sin', 'cos', 'sin_abs', 'sin_squared']:
            amp = np.random.uniform(0.5, 5.0)
            freq = np.random.uniform(0.1, 0.5)
            phase = np.random.uniform(0, 2*np.pi)
            shift = np.random.uniform(-3.0, 3.0)
            params = [amp, freq, phase, shift]
        elif self.func_type == 'sin_cos':
            amp_sin = np.random.uniform(0.5, 3.0)
            amp_cos = np.random.uniform(0.5, 3.0)
            freq = np.random.uniform(0.1, 0.5)
            shift = np.random.uniform(-3.0, 3.0)
            params = [amp_sin, amp_cos, freq, shift]
        return params


    def evaluate(self, params_list, x):
        x = np.asarray(x).flatten()
        if self.func_type == 'sin':
            amp, freq, phase, shift = params_list
            return amp * np.sin(freq * x + phase) + shift
        elif self.func_type == 'cos':
            amp, freq, phase, shift = params_list
            return amp * np.cos(freq * x + phase) + shift
        elif self.func_type == 'sin_abs':
            amp, freq, phase, shift = params_list
            return amp * np.abs(np.sin(freq * x + phase)) + shift
        elif self.func_type == 'sin_squared':
            amp, freq, phase, shift = params_list
            return amp * (np.sin(freq * x + phase))**2 + shift
        elif self.func_type == 'sin_cos':
            amp_sin, amp_cos, freq, shift = params_list
            return amp_sin * np.sin(freq * x) + amp_cos * np.cos(freq * x) + shift

    def __call__(self, x):
        return self.evaluate(self.true_params, x)

    def get_true_params(self):
        return self.true_params.copy()
    


'===== КЛАСС КУСОЧНО-ЛИНЕЙНЫХ 1D ФУНКЦИЙ ====='
class Piecewise:
    def __init__(self, func_type=None):
        self.types = [
            'linear_sin', 'linear_cos', 'quadratic_linear',
            'nonlinear_linear_nonlinear', 'linear_nonlinear_linear'
        ]
        if func_type is None:
            self.func_type = np.random.choice(self.types)
        else:
            self.func_type = func_type

        self.params_dict = self._generate_random_params()  # словарь со всеми параметрами, чтобы понимать какая конкрентно функция справа в середине и слева
        self.param_names = [key for key, value in self.params_dict.items() #все ключи у которых значение - не строка, чтобы в evaluate преобразовать
                            #прешедший из функции оптимизаии массив в словарь и в формулу подставлять параметры в виде p[правый_сдвиг](для наглядности)
                           if not isinstance(value, str)]
        self.true_params = [self.params_dict[name] for name in self.param_names]#тут только числа - параметры, чтобы вызвать с ними evaluate в генераторах

        # Устанавливаем точки разрыва
        if self.func_type in ['linear_sin', 'linear_cos', 'quadratic_linear']:
            self.break_point = np.random.uniform(-3, 3)
        elif self.func_type in ['nonlinear_linear_nonlinear', 'linear_nonlinear_linear']:
            b1 = np.random.uniform(-5, 0)
            b2 = np.random.uniform(0, 5)
            self.break1 = b1
            self.break2 = b2

    def _generate_random_params(self):
        params = {}
        if self.func_type == 'linear_sin':
            params['a_left'] = np.random.uniform(-0.2, 2.0)
            params['b_left'] = np.random.uniform(-2, 2)
            params['amp_right'] = np.random.uniform(0.5, 3.0)
            params['freq_right'] = np.random.uniform(0.1, 0.5)
            params['phase_right'] = np.random.uniform(0, 2*np.pi)
            params['shift_right'] = np.random.uniform(-2, 2)
        elif self.func_type == 'linear_cos':
            params['a_left'] = np.random.uniform(-0.2, 2.0)
            params['b_left'] = np.random.uniform(-2, 2)
            params['amp_right'] = np.random.uniform(0.5, 3.0)
            params['freq_right'] = np.random.uniform(0.1, 0.5)
            params['phase_right'] = np.random.uniform(0, 2*np.pi)
            params['shift_right'] = np.random.uniform(-2, 2)
        elif self.func_type == 'quadratic_linear':
            params['a_left'] = np.random.uniform(-0.2, 2.0)
            params['b_left'] = np.random.uniform(-2, 2)
            params['c_left'] = np.random.uniform(-2, 2)
            params['a_right'] = np.random.uniform(0.2, 2.0)
            params['b_right'] = np.random.uniform(-2, 2)
        elif self.func_type == 'nonlinear_linear_nonlinear':
            left_type = np.random.choice(['quadratic', 'sin', 'cos'])
            if left_type == 'quadratic':
                params['left_a'] = np.random.uniform(0.2, 2.0)
                params['left_b'] = np.random.uniform(-2, 2)
                params['left_c'] = np.random.uniform(-2, 2)
                params['left_type'] = 'quadratic'
            elif left_type == 'sin':
                params['left_amp'] = np.random.uniform(0.5, 3.0)
                params['left_freq'] = np.random.uniform(0.1, 0.5)
                params['left_phase'] = np.random.uniform(0, 2*np.pi)
                params['left_shift'] = np.random.uniform(-2, 2)
                params['left_type'] = 'sin'
            elif left_type == 'cos':
                params['left_amp'] = np.random.uniform(0.5, 3.0)
                params['left_freq'] = np.random.uniform(0.1, 0.5)
                params['left_phase'] = np.random.uniform(0, 2*np.pi)
                params['left_shift'] = np.random.uniform(-2, 2)
                params['left_type'] = 'cos'
            params['mid_a'] = np.random.uniform(0.2, 2.0)
            params['mid_b'] = np.random.uniform(-2, 2)
            right_type = np.random.choice(['quadratic', 'sin', 'cos'])
            if right_type == 'quadratic':
                params['right_a'] = np.random.uniform(0.2, 2.0)
                params['right_b'] = np.random.uniform(-2, 2)
                params['right_c'] = np.random.uniform(-2, 2)
                params['right_type'] = 'quadratic'
            elif right_type == 'sin':
                params['right_amp'] = np.random.uniform(0.5, 3.0)
                params['right_freq'] = np.random.uniform(0.1, 0.5)
                params['right_phase'] = np.random.uniform(0, 2*np.pi)
                params['right_shift'] = np.random.uniform(-2, 2)
                params['right_type'] = 'sin'
            elif right_type == 'cos':
                params['right_amp'] = np.random.uniform(0.5, 3.0)
                params['right_freq'] = np.random.uniform(0.1, 0.5)
                params['right_phase'] = np.random.uniform(0, 2*np.pi)
                params['right_shift'] = np.random.uniform(-2, 2)
                params['right_type'] = 'cos'
        elif self.func_type == 'linear_nonlinear_linear':
            params['left_a'] = np.random.uniform(0.2, 2.0)
            params['left_b'] = np.random.uniform(-2, 2)
            mid_type = np.random.choice(['quadratic', 'sin', 'cos'])
            if mid_type == 'quadratic':
                params['mid_a'] = np.random.uniform(0.2, 2.0)
                params['mid_b'] = np.random.uniform(-2, 2)
                params['mid_c'] = np.random.uniform(-2, 2)
                params['mid_type'] = 'quadratic'
            elif mid_type == 'sin':
                params['mid_amp'] = np.random.uniform(0.5, 3.0)
                params['mid_freq'] = np.random.uniform(0.1, 0.5)
                params['mid_phase'] = np.random.uniform(0, 2*np.pi)
                params['mid_shift'] = np.random.uniform(-2, 2)
                params['mid_type'] = 'sin'
            elif mid_type == 'cos':
                params['mid_amp'] = np.random.uniform(0.5, 3.0)
                params['mid_freq'] = np.random.uniform(0.1, 0.5)
                params['mid_phase'] = np.random.uniform(0, 2*np.pi)
                params['mid_shift'] = np.random.uniform(-2, 2)
                params['mid_type'] = 'cos'
            params['right_a'] = np.random.uniform(0.2, 2.0)
            params['right_b'] = np.random.uniform(-2, 2)
        return params

    def evaluate(self, params_list, x):
        """params_list: список значений в порядке self.param_names"""
        # Преобразуем список в словарь для удобства использования и наглядности
        p = dict(zip(self.param_names, params_list))
        
        x = np.asarray(x).flatten()
        y = np.zeros_like(x)

        if self.func_type in ['linear_sin', 'linear_cos', 'quadratic_linear']:
            #деление точек на до точки до разрыва + разрыв и после
            left_mask = x <= self.break_point #создаёт массив - маску, в которой True для точек, которые меньше или равны точке разрыва, и False для остальных
            right_mask = ~left_mask#меняет в маске True на False и наоборот

            if self.func_type == 'linear_sin':
                y[left_mask] = p['a_left'] * x[left_mask] + p['b_left']
                y[right_mask] = (p['amp_right'] * np.sin(p['freq_right'] * x[right_mask] + p['phase_right'])+ p['shift_right'])
    
            elif self.func_type == 'linear_cos':
                y[left_mask] = p['a_left'] * x[left_mask] + p['b_left']
                y[right_mask] = (p['amp_right'] * np.cos(p['freq_right'] * x[right_mask] + p['phase_right'])
                                 + p['shift_right'])
            elif self.func_type == 'quadratic_linear':
                y[left_mask] = p['a_left'] * (x[left_mask] - p['b_left'])**2 + p['c_left']
                y[right_mask] = p['a_right'] * x[right_mask] + p['b_right']

        elif self.func_type == 'nonlinear_linear_nonlinear':
            left_mask = x <= self.break1
            mid_mask = (x > self.break1) & (x <= self.break2)
            right_mask = x > self.break2

            lt = self.params_dict['left_type']
            if lt == 'quadratic':
                y[left_mask] = p['left_a'] * (x[left_mask] - p['left_b'])**2 + p['left_c']
            elif lt == 'sin':
                y[left_mask] = p['left_amp'] * np.sin(p['left_freq'] * x[left_mask] + p['left_phase']) + p['left_shift']
            elif lt == 'cos':
                y[left_mask] = p['left_amp'] * np.cos(p['left_freq'] * x[left_mask] + p['left_phase']) + p['left_shift']

            y[mid_mask] = p['mid_a'] * x[mid_mask] + p['mid_b']

            rt = self.params_dict['right_type']
            if rt == 'quadratic':
                y[right_mask] = p['right_a'] * (x[right_mask] - p['right_b'])**2 + p['right_c']
            elif rt == 'sin':
                y[right_mask] = p['right_amp'] * np.sin(p['right_freq'] * x[right_mask] + p['right_phase']) + p['right_shift']
            elif rt == 'cos':
                y[right_mask] = p['right_amp'] * np.cos(p['right_freq'] * x[right_mask] + p['right_phase']) + p['right_shift']

        elif self.func_type == 'linear_nonlinear_linear':
            left_mask = x <= self.break1
            mid_mask = (x > self.break1) & (x <= self.break2)
            right_mask = x > self.break2

            y[left_mask] = p['left_a'] * x[left_mask] + p['left_b']

            mt = self.params_dict['mid_type']
            if mt == 'quadratic':
                y[mid_mask] = p['mid_a'] * (x[mid_mask] - p['mid_b'])**2 + p['mid_c']
            elif mt == 'sin':
                y[mid_mask] = p['mid_amp'] * np.sin(p['mid_freq'] * x[mid_mask] + p['mid_phase']) + p['mid_shift']
            elif mt == 'cos':
                y[mid_mask] = p['mid_amp'] * np.cos(p['mid_freq'] * x[mid_mask] + p['mid_phase']) + p['mid_shift']

            y[right_mask] = p['right_a'] * x[right_mask] + p['right_b']

        return y

    def __call__(self, x):
        return self.evaluate(self.true_params, x)

    def get_true_params(self):
        return self.true_params.copy()


'====КЛАССЫ - ГЕНЕРАТОРЫ===='

class UnimodalGenerator:
    def __init__(self, x_vals):
        self.x_vals = x_vals

    def generate(self, n_functions=15):
        functions = []
        for i in range(n_functions):
            func = Unimodal()
            y = func(self.x_vals)#это получилось блягодаря методу call
            functions.append({
                'class': 'unimodal',
                'type': func.func_type,
                'true_params': func.get_true_params(),
                'func_obj': func,
                'x': self.x_vals.copy(),
                'y': y,
                'id':i
            })
        return functions
  
class PeriodicGenerator:
    def __init__(self, x_vals):
        self.x_vals = x_vals

    def generate(self, n_functions=15):
        functions = []
        for i in range(n_functions):
            func = Periodic()
            y = func(self.x_vals)
            functions.append({
                'class': 'periodic',
                'type': func.func_type,
                'true_params': func.get_true_params(),
                'func_obj': func,
                'x': self.x_vals.copy(),
                'y': y,
                'id':i
            })
        return functions


class PiecewiseGenerator:
    def __init__(self, x_vals):
        self.x_vals = x_vals

    def generate(self, n_functions=15):
        functions = []
        for i in range(n_functions):
            func = Piecewise()
            y = func(self.x_vals)
            data = {
                'class': 'piecewise',
                'type': func.func_type,
                'true_params': func.get_true_params(),
                'func_obj': func,
                'x': self.x_vals.copy(),
                'y': y,
                'id':i
            }
            if hasattr(func, 'break_point'):
                data['break_point'] = func.break_point
            if hasattr(func, 'break1') and hasattr(func, 'break2'):
                data['break1'] = func.break1
                data['break2'] = func.break2
            functions.append(data)
        return functions
    


'====Создание и сохранение всех функций из класса===='

def plot_and_save_functions(data, class_name, filename, n_cols=5):
    n_funcs = len(data)
    n_rows = (n_funcs + n_cols - 1) // n_cols #сколько строк для размещения всех графиков
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))#создает сетку графиков
    axes = axes.flatten()#из двумерного в одномерный для удобства обращения по индексу
    for i, item in enumerate(data):
        axes[i].plot(item['x'], item['y'])#рисует i-ый график на i-ой оси
        axes[i].set_title(f"{item['type']}")#заголовок для i-ого графика
        axes[i].grid(True)
        # Добавляем небольшой отступ по Y для лучшей видимости
        y_min, y_max = min(item['y']), max(item['y'])
        y_range = y_max - y_min
        axes[i].set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)#устанавливвает границы с учетом отступов для графика
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)#скрывает неиспоьзованные оси если такие есть
    plt.suptitle(f"Класс: {class_name}")
    plt.tight_layout()#отступы между графиками
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"График сохранен в {filename}")
    


'==== Main ===='
if __name__ == "__main__":

    # Загружаем существующие x_vals или создаем новые
    if os.path.exists('x_vals.npy'):
        x_vals = np.load('x_vals.npy')
        print(f"Загружены x_vals из файла, размер: {len(x_vals)}")
    else:
        x_vals = np.linspace(-16, 16, 100)
        np.save('x_vals.npy', x_vals)

    unimodal_gen = UnimodalGenerator(x_vals)
    periodic_gen = PeriodicGenerator(x_vals)
    piecewise_gen = PiecewiseGenerator(x_vals)
    
    # Генерируем по 15 новых функций каждого класса
    print("\nГенерируем новые функции...")
    unimodal = unimodal_gen.generate(25)
    periodic = periodic_gen.generate(25)
    piecewise = piecewise_gen.generate(25)
    
    print(f"Сгенерировано:")
    print(f"  - Унимодальных: {len(unimodal)}")
    print(f"  - Периодических: {len(periodic)}")
    print(f"  - Кусочных: {len(piecewise)}")
    
    all_train_data = {
        'unimodal': unimodal,
        'periodic': periodic,
        'piecewise': piecewise
    }

    with open('train_functions.pkl', 'wb') as f:
        pickle.dump(all_train_data, f)
    print("\nДанные сохранены в train_functions.pkl")
    
    print("\nСоздание и сохранение графиков...")
    
    os.makedirs('StartFunctionsPlots(25*3)', exist_ok=True)
    
    # Сохраняем графики для каждого класса
    plot_and_save_functions(unimodal, 'Unimodal', 'StartFunctionsPlots(25*3)/unimodal.png')
    plot_and_save_functions(periodic, 'Periodic', 'StartFunctionsPlots(25*3)/periodic.png')
    plot_and_save_functions(piecewise, 'Piecewise', 'StartFunctionsPlots(25*3)/piecewise.png')
    
    print("\nГрафики сохранены в директории 'StartFunctionsPlots(25*3)/':")
    print("  - /unimodal.png")
    print("  - /periodic.png")
    print("  - /piecewise.png")