from scipy.stats import qmc
from scipy.stats import kstest
from scipy.stats import qmc
from GenerateFunc_1 import Unimodal, Periodic, Piecewise,plot_and_save_functions
from scipy.optimize import differential_evolution
import os
import numpy as np
import pickle
import time


distributions = {
    "Нормальное": 'norm',
    "Равномерное": 'uniform',
    "Экспоненциальное": 'expon',
    "Гамма": 'gamma',
    "Бета": 'beta',
    "Хи-квадрат": 'chi2',
    "Стьюдента (t)": 't',
    "Логнормальное": 'lognorm',
    "Вейбулла": 'weibull_min',
    "Коши": 'cauchy',
    "Парето": 'pareto',
    "Лапласа": 'laplace',
    "Логистическое": 'logistic',
    "Рэлея": 'rayleigh'
}

def hyperparameter_tuning(x, params, func_obj, distribution, nfev, n_samples=128,bounds=None):
    """Поиск гиперпараметров - возвращает топ-5% результатов"""
    n_points = len(x)
    param_bounds = {
        'min_point': (int(0.1*n_points), int(0.7*n_points)),
        'step': (10, 60),
        'coeff': (0.1, 0.9),
        'stable_iterations': (int(0.01*nfev), int(0.1*nfev))
    }

    #Перебирать все параметры в циклах - долго, поэтому используем последовательность Соболя
    #И масштабируем под наши геперпараметры точки из последовательности Соболя
    sampler = qmc.Sobol(d=len(param_bounds), scramble=True)
    samples = sampler.random(n_samples)

    results = []

    for i, sample in enumerate(samples):
        #Масштабтрование
        config = {
            'min_point': int(sample[0] * (param_bounds['min_point'][1] - param_bounds['min_point'][0]) + param_bounds['min_point'][0]),
            'step': int(sample[1] * (param_bounds['step'][1] - param_bounds['step'][0]) + param_bounds['step'][0]),
            'coeff': round(sample[2] * (param_bounds['coeff'][1] - param_bounds['coeff'][0]) + param_bounds['coeff'][0], 1),
            'stable_iterations': int(sample[3] * (param_bounds['stable_iterations'][1] - param_bounds['stable_iterations'][0]) + param_bounds['stable_iterations'][0])
        }

        problem = Model_for_search(
            x=x,
            params=params,
            **config,
            func=func_obj,
            delt=1,
            rasp=distribution
        )
        problem.calc_and_save_func()
        
        n_params = len(params)
        #Границы параметров передаются в функцию - верные границы это важно для периодического класса функций
        if bounds is None:
            if len(params) == 3:
                bounds = [(-20, 20), (-15, 15), (-20, 20)]
            else:
                bounds = [(-20, 20)] * n_params

        # Дифференциальная эволюция 
        res = differential_evolution(
            problem.total_residual,
            bounds,
            maxiter=200,            
            popsize=10,              # меньшая популяция для скорости
            seed=0,
            tol=1e-8,
            updating='deferred',
            workers=1,
            disp=False
        )
        
        param_error = np.mean(np.abs(res.x - params))

        # Сохраняем результат
        results.append({
            'error': param_error,
            'params_found': res.x.copy(),
            'hyperparams': config,
            'func_calls': problem.counter,
            'total_res_calls': res.nfev,
            'remaining_points_count': len(problem.get_x()),
            'remaining_points': problem.get_x().copy()
        })

        if (i + 1) % 100 == 0:
            print(f"  tuning progress: {i+1}/{n_samples}")

    # Сортируем по ошибке
    results.sort(key=lambda x: x['error'])
    five_percent = max(1, int(len(results) * 0.05))
    best_results = results[:five_percent]

    return best_results  # возвращаем и топ-5%


class Model_for_search:
    def __init__(self, x, func, params, step, min_point, rasp, coeff, stable_iterations, delt): #Конструктор
        self.__x = np.copy(x)  # вектор точек
        self.func = func  # рассматриваемая функция
        self.calc_func_in_x = []  # значение функции в точке эталон
        self.__params = params  # параметры эталон
        self.step = step  # шаг
        self.iter = 0  # количество итераций
        self.raspred = rasp  # параметр для распределения
        # сколько итераций ждать чтобы параметры стабилизировались
        self.stable_iterations = stable_iterations
        self.coeff = coeff
        self.curr_dim = self.__x.shape[0]  # текущая размерность
        self.counter = 0  # количество вызовов функции для подсчёта ошибки
        self.delt = delt  # флаг- посмотреть результат с удалением или без него
        # self.need_calc = need_calc  # ограничение по итерациям
        self.min_point_c = min_point
        # минимальное количество точек после удаления

    
    def Check_Rasp(self, i): #Проверка распределения
        # скопируем текущий набор, удалим из копии i-ую точку, проверим распределление
        copy_current_x = np.copy(self.__x)
        copy_current_x = np.delete(copy_current_x, i, axis=0).flatten()

        if self.raspred == 'uniform':
            # Для равномерного распределения указываем границы данных
            a, b = np.min(copy_current_x), np.max(copy_current_x)
            stat, p_value = kstest(copy_current_x, 'uniform', args=(a, b - a))
        else:
            # Для остальных распределений (нормального, экспоненциального и т. д.)
            stat, p_value = kstest(copy_current_x, self.raspred)

        return p_value > 0.05  # Если p-value > 0.05, распределение не нарушено

    def calc_and_save_func(self):  #подсчёт  i-ой точке i-ое значение функции
        for i in range(len(self.__x)):
            func_in_x = self.func.evaluate(self.__params, self.__x[i])
            self.calc_func_in_x.append(func_in_x)

    def vector_residual(self, params, i):  # отклонение для i-ой точки в наборе
        new_res = self.func.evaluate(params, self.__x[i])
        self.counter += 1
        return (new_res - self.calc_func_in_x[i])**2

    def total_residual(self, params):  # отклонение для всего набора
        total_error = 0
        errors = []
        self.iter += 1
        for i in range(len(self.__x)):
            res_for_vector_i = self.vector_residual(params, i)
            errors.append(res_for_vector_i)
            if res_for_vector_i != float('inf'):
                total_error += res_for_vector_i  # Суммируем ошибки
        if (self.delt):
            self.Del_point(total_error, errors)
        return (total_error/self.curr_dim)

    

    def Del_point(self, total_error, errors):  # функция удаления
        if (self.iter > self.stable_iterations and len(self.__x) > self.min_point_c and self.iter % self.step != 0):
            contributions = []
            for i in range(len(self.__x)):
                # Вычисляем ошибку без текущей точки
                error_without_point = total_error - errors[i]
                contribution = abs(total_error - error_without_point) / \
                    total_error if total_error != 0 else 0
                # массив ошибок без iой точки
                contributions.append(contribution)
            min_contribution = min(contributions)
            eps_threshold = np.mean(contributions) * self.coeff
            if min_contribution >= eps_threshold:
                return  # Нет точек для удаления
            # Индекс точки с мин. вкладом
            to_remove = np.argmin(contributions)
            if (self.Check_Rasp(to_remove)):
                self.curr_dim -= 1
                self.__x = np.delete(self.__x, to_remove, axis=0)

    def print_x(self):
        for i in range(self.__x.shape[0]):
            print(self.__x[i])

    def get_func_in_x(self):
        return self.calc_func_in_x

    def get_x(self):
        return self.__x

    def set_raspr(self, raspr):
        self.raspred = raspr

def process_single_function(func_data, class_name, base_dir='results'):
    """
    func_data: словарь с ключами 'func_obj', 'x', 'y', 'params','id' (эталонные параметры)
    class_name: 'unimodal', 'periodic', 'piecewise'
    base_dir: базовая директория для сохранения

    Тут запускаем подбор параметров без удалания, потом поиск геперпараметров при удалении, сохраняем 
    5% лучших результатов без удаления в файл и возвращаем лучший резултат для сохранения
    """
    # Создаём папку для класса
    class_dir = os.path.join(base_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    id = func_data['id']
    # Для каждого класса своя папка, внутри 15 txt файлов
    txt_filename = os.path.join(class_dir, f'func_{id:02d}_results.txt')

    func_obj = func_data['func_obj']
    x = func_data['x']
    true_params = func_data['true_params']
    y = func_data['y']

    #Оптимизация без удаления
    print(f"Processing {class_name} function {id} without deletion...")
    problem_no_del = Model_for_search(
        x=x,
        params=true_params,
        func=func_obj,
        step=0,
        min_point=0,
        rasp='',
        coeff=0,
        stable_iterations=0,
        delt=0
    )
    problem_no_del.calc_and_save_func()
    
    # Определяем границы для каждого типа функций
    if class_name == 'unimodal':
        if func_data.get('type') == 'gaussian':
            # Для гауссианы: a, mu, sigma (>0), c
            bounds = [(-20, 20), (-15, 15), (0.1, 10), (-20, 20)]
        else:
            # Для quadratic и absolute
            bounds = [(-20, 20), (-15, 15), (-20, 20)]
    elif class_name == 'periodic':
        #freq - важный параметр который влияет на остальные
        if func_data.get('type') == 'sin_cos':
            # amp_sin, amp_cos, freq, shift
            bounds = [(-10, 10), (-10, 10), (0.1, 0.5), (-10, 10)]
        else:
            # amp, freq, phase, shift
            bounds = [(-10, 10), (0.1, 0.5), (0, 2*np.pi), (-10, 10)]
    elif class_name == 'piecewise':
        bounds = []
        for i, name in enumerate(func_obj.param_names):
            if 'freq' in name.lower():
                bounds.append((0.1, 0.5))        # частота freq - важный параметр который влияет на остальные
            elif 'phase' in name.lower():
                bounds.append((0, 2*np.pi))      # фаза по кругу
            else:
                bounds.append((-10, 10))         # всё остальное не важно

    print(f"  Запуск differential_evolution...")

    # Запускаем дифференциальную эволюцию
    res_no_del = differential_evolution(
        problem_no_del.total_residual,
        bounds,
        maxiter=200,           # максимальное количество поколений  - было 1000, для скорости 200
        popsize=15,             # размер популяции
        seed=0,                  # для воспроизводимости
        tol=1e-10,               # допуск по функции
        updating='deferred',     # стратегия обновления
        workers=1,               # количество процессов (1 = без параллелизации)
        disp=False               # не выводить прогресс
    )

    nfev = res_no_del.nfev
    params_no_del = res_no_del.x
    print(f"  эталонные параметры: {true_params}")
    print(f"  найденные параметры: {params_no_del}")
    print(f"  ошибка: {np.mean(np.abs(params_no_del - true_params)):.6f}")
    print(f"  nfev: {nfev}")
    
    # Получаем топ-5% результатов с удалением
    print(f"  запуск hyperparameter_tuning...")
    best_results = hyperparameter_tuning(
        x,
        true_params,
        func_obj,
        distribution='uniform',
        nfev=nfev,
        bounds=bounds
    )

    # Сохраняем топ-5% в txt файл
    with open(txt_filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"RESULTS FOR {class_name} FUNCTION {id}\n")
        f.write("="*80 + "\n\n")
        
        f.write("ORIGINAL DATA:\n")
        f.write(f"  True parameters: {true_params}\n")
        f.write(f"  Number of points: {len(x)}\n")
        f.write(f"  X range: [{min(x):.2f}, {max(x):.2f}]\n\n")
        
        f.write("WITHOUT DELETION:\n")
        f.write(f"  Found parameters: {params_no_del}\n")
        f.write(f"  Error: {np.mean(np.abs(params_no_del - true_params)):.6f}\n")
        f.write(f"  nfev: {nfev}\n\n")
        
        f.write("="*80 + "\n")
        f.write(f"TOP 5% RESULTS WITH DELETION (total {len(best_results)} results)\n")
        f.write("="*80 + "\n\n")
        
        for i, res in enumerate(best_results):
            f.write(f"--- Result {i+1} ---\n")
            f.write(f"  Error: {res['error']:.6f}\n")
            f.write(f"  Found parameters: {res['params_found']}\n")
            f.write(f"  Hyperparameters:\n")
            for k, v in res['hyperparams'].items():
                f.write(f"    {k}: {v}\n")
            f.write(f"  Function calls: {res['func_calls']}\n")
            f.write(f"  Total residual calls: {res['total_res_calls']}\n")
            f.write(f"  Remaining points: {res['remaining_points_count']}\n")
            f.write(f"  Remaining points values: {res['remaining_points'][:10]}... (first 10)\n\n")
    
    print(f"  сохранено в {txt_filename}")
    best_result = best_results[0]
    
    return {
        'params_del': best_result['params_found'],  # найденные параметры
        'x_remaining': best_result['remaining_points'],  # оставшиеся точки
        'hyperparams': best_result['hyperparams'],
        'error': best_result['error'],
        'func_calls': best_result['func_calls'],
        'params_no_del':params_no_del
    }



def process_all_functions(data_file='train_functions.pkl', base_dir = 'del_results', resf = 'del_results/all_results.pkl'):
    # Загружаем данные
    with open(data_file, 'rb') as f:
        all_data = pickle.load(f)
    
    if os.path.exists(resf):
        with open(resf, 'rb') as fl:
            all_results = pickle.load(fl)
        print(f"Загружены существующие результаты из {resf}")
        print(f"  unimodal: {len(all_results.get('unimodal', []))} функций")
        print(f"  periodic: {len(all_results.get('periodic', []))} функций")
        print(f"  piecewise: {len(all_results.get('piecewise', []))} функций")
    else:
        all_results = {
            'unimodal': [],
            'periodic': [],
            'piecewise': []
        }

    # Обрабатываем каждый класс
    for class_name, func_list in all_data.items():
        print(f"\nProcessing {class_name} functions...")
        for func_data in func_list:
            id = func_data['id']
            txt_filename = os.path.join(base_dir, class_name, f'func_{id:02d}_results.txt')
            if os.path.exists(txt_filename):
                print(f"  Function {id}/{len(func_list)} уже обработана, пропускаем")
                continue
            print(f"  Function {id}/{len(func_list)}")
            
            # Получаем результаты для одной функции
            best_del_result = process_single_function(func_data, class_name, base_dir)
            #тут соединяем данные лучшего удаления и данные из train_functions.pkl
            combined_data = {**func_data, **best_del_result}
            
            all_results[class_name].append(combined_data)
            with open(resf, 'wb') as fi:
                pickle.dump(all_results, fi)
                
            # Небольшая задержка
            time.sleep(1)


if __name__ == "__main__":
    process_all_functions('train_functions.pkl', 'del_results')