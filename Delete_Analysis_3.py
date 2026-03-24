import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random

def create_save_4gr(original_data, no_del_data, del_data, recovered_data, class_name, func_id, save_dir='plots'):
    """
    Создает и сохраняет 4 графика на одной картинке
    
    Args:
        original_data: (x, y) исходной функции
        no_del_data: (x, y) функции с параметрами без удаления
        del_data: (x_remaining, y_remaining) оставшиеся после удаления точки
        recovered_data: (x, y_recovered) восстановленная функция по параметрам после удаления
        class_name: название класса
        func_id: id функции
        save_dir: директория для сохранения
    """
    x_orig, y_orig = original_data
    x_no_del, y_no_del = no_del_data
    x_rem, y_rem = del_data
    x_rec, y_rec = recovered_data
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # График 1 - Исходная функция
    axes[0, 0].plot(x_orig, y_orig, 'b-', linewidth=2, label='Исходная')
    axes[0, 0].set_title('1. Исходная функция')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].legend()
    
    # График 2 - Функция с параметрами без удаления
    axes[0, 1].plot(x_no_del, y_no_del, 'g-', linewidth=2, label='Без удаления')
    axes[0, 1].plot(x_orig, y_orig, 'b--', linewidth=1, alpha=0.5, label='Исходная')
    axes[0, 1].set_title('2. Без удаления')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].legend()
    
    # График 3 - Восстановленная функция после удаления (все точки)
    axes[1, 0].plot(x_rec, y_rec, 'r-', linewidth=2, label='Восстановленная (после удаления)')
    axes[1, 0].plot(x_orig, y_orig, 'b--', linewidth=1, alpha=0.5, label='Исходная')
    axes[1, 0].set_title('3. Восстановленная функция (по всем x)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].legend()
    
    # График 4 - Восстановленная функция с выделенными оставшимися точками
    axes[1, 1].plot(x_rec, y_rec, 'r-', linewidth=2, alpha=0.7, label='Восстановленная')
    axes[1, 1].scatter(x_rem, y_rem, c='red', s=50, marker='o', zorder=5, 
                       label=f'Оставшиеся точки ({len(x_rem)} шт.)')
    axes[1, 1].plot(x_orig, y_orig, 'b--', linewidth=1, alpha=0.5, label='Исходная')
    axes[1, 1].set_title(f'4. Оставшиеся точки на восстановленной кривой')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].legend()
    
    # Общий заголовок
    plt.suptitle(f'{class_name} function {func_id}', fontsize=14)
    plt.tight_layout()
    
    # Сохранение
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f'{class_name}_func_{func_id:02d}_comparison.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  График сохранен: {filename}")



def plot_random_functions_from_results(data_file='del_results/best_del_results.pkl', 
                                        n_samples=3, 
                                        save_dir='comparison_plots_res'):
    """
    Из файла результатов берет по 3 случайные функции из каждого класса
    и строит для них сравнительные графики
    """
    # Загружаем данные
    with open(data_file, 'rb') as f:
        all_results = pickle.load(f)
    
    print("=" * 60)
    print("Создание сравнительных графиков для случайных функций")
    print("=" * 60)
    
    for class_name in ['unimodal', 'periodic', 'piecewise']:
        func_list = all_results.get(class_name, [])
        
        if not func_list:
            print(f"\n{class_name}: нет данных")
            continue
        
        # Берем случайные функции
        n_to_sample = min(n_samples, len(func_list))
        sampled_funcs = random.sample(func_list, n_to_sample)
        
        print(f"\n{class_name}: выбрано {n_to_sample} функций")
        
        for func_data in sampled_funcs:
            func_id = func_data.get('id', 0)
            
            #Исходные данные
            x_orig = func_data['x']
            y_orig = func_data['y']
            
            #Данные без удаления
            params_no_del = func_data.get('params_no_del')
            func_obj = func_data['func_obj']
            y_no_del = func_obj.evaluate(params_no_del, x_orig)
            
            #Данные после удаления
            params_del = func_data.get('params_del')
            x_remaining = func_data.get('x_remaining')
            # y для оставшихся точек
            y_rem = func_obj.evaluate(params_del, x_remaining)
            
            #Восстановленная функция по всем x_orig (с параметрами после удаления)
            y_recovered = func_obj.evaluate(params_del, x_orig)
            
            #Создаем и сохраняем график
            create_save_4gr(
                original_data=(x_orig, y_orig),
                no_del_data=(x_orig, y_no_del),
                del_data=(x_remaining, y_rem),
                recovered_data=(x_orig, y_recovered),
                class_name=class_name,
                func_id=func_id,
                save_dir=save_dir
            )
    
    print(f"\nВсе графики сохранены в директории '{save_dir}/'")




def compute_mean_hyperparams(data_file='del_results/best_del_results.pkl', 
                              output_file='mean_hyper_Params.pkl'):
    """
    Вычисляет средние значения гиперпараметров для каждого класса
    и сохраняет в файл
    """
    # Загружаем данные
    with open(data_file, 'rb') as f:
        all_results = pickle.load(f)
    
    mean_hyperparams = {}
    
    print("\n" + "=" * 60)
    print("Вычисление средних гиперпараметров по классам")
    print("=" * 60)
    
    for class_name in ['unimodal', 'periodic', 'piecewise']:
        func_list = all_results.get(class_name, [])
        
        # Собираем все гиперпараметры
        min_points = []
        steps = []
        coeffs = []
        stable_iters = []
        
        for func_data in func_list:
            hyperparams = func_data.get('hyperparams')
            if hyperparams:
                min_points.append(hyperparams.get('min_point'))
                steps.append(hyperparams.get('step'))
                coeffs.append(hyperparams.get('coeff'))
                stable_iters.append(hyperparams.get('stable_iterations'))
        
        mean_hyperparams[class_name] = {
            'min_point': np.mean(min_points),
            'step': np.mean(steps),
            'coeff': np.mean(coeffs),
            'stable_iterations': np.mean(stable_iters)
        }
        
        print(f"\n{class_name}:")
        print(f"  min_point: {mean_hyperparams[class_name]['min_point']:.2f}")
        print(f"  step: {mean_hyperparams[class_name]['step']:.2f}")
        print(f"  coeff: {mean_hyperparams[class_name]['coeff']:.3f}")
        print(f"  stable_iterations: {mean_hyperparams[class_name]['stable_iterations']:.2f}")
        print(f"  (на основе {len(min_points)} функций)")
        
    
    # Сохраняем в файл
    with open(output_file, 'wb') as f:
        pickle.dump(mean_hyperparams, f)
    
    print(f"\nСредние гиперпараметры сохранены в {output_file}")
    

if __name__ == "__main__":
    # 1. Создаем сравнительные графики для случайных функций
    plot_random_functions_from_results(
        data_file='del_results/best_del_results.pkl',
        n_samples=3,
        save_dir='comparison_plots_res'
    )
    
    # 2. Вычисляем и сохраняем средние гиперпараметры
    compute_mean_hyperparams(
        data_file='del_results/best_del_results.pkl',
        output_file='Mean_HyperP.pkl'
    )
    