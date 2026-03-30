import numpy as np
import pickle
from GenerateFunc_1 import Unimodal, Periodic, Piecewise
from Delete_Analysis_3 import create_save_4gr
from FindParams_Delete_2 import Model_for_search
from scipy.optimize import differential_evolution

if __name__ == "__main__":
   

    # Загружаем функции, которые мы верно классифицировали
    with open('correctly_classified.pkl', 'rb') as f:
        functions = pickle.load(f)

    count = min(len(functions['unimodal']),len(functions['periodic']),len(functions['piecewise']),5)
    classes = ['unimodal', 'periodic', 'piecewise']
    for j in range(len(classes)):
        for i in range(count):
            func_data =  functions[classes[j]][i]
            x_orig = func_data.get('x')
            y_orig = func_data.get('y')
            func = func_data.get('func_obj')
            params = func_data.get('true_params')
            problem_no_del = Model_for_search(
                x=x_orig,
                params=params,
                func=func,
                step=0,
                min_point=0,
                rasp='',
                coeff=0,
                stable_iterations=0,
                delt=0
            )
            problem_no_del.calc_and_save_func()
            if classes[j] == 'unimodal':
                if func_data.get('type') == 'gaussian':
                    # Для гауссианы: a, mu, sigma (>0), c
                    bounds = [(1, 20), (-15, 15), (0.1, 10), (-20, 20)]
                elif func_data.get('type') == 'quadratic' or func_data.get('type') =='absolute':
                    bounds = [(-20, 20), (-15, 15), (-20, 20)]
                elif func_data.get('type') == 'quadratic_log':
                    bounds = [(-20, 20), (-15, 15), (-20, 20),(0.1, 2.0), (-10,10)]
                elif func_data.get('type') == 'quadratic_exp':
                    bounds = [(-20, 20), (-15, 15), (-20, 20),(0.1, 1.5), (-10,10)]
                elif func_data.get('type') == 'gaussian_poly':
                    bounds = [(0, 20), (-15, 15), (0.5, 3.0),(0.1, 1.5), (0.1, 1.0)]
                

            elif classes[j] == 'periodic':
                #freq - важный параметр который влияет на остальные
                
                if func_data.get('type') == 'sin_cos_sin':
                   bounds = [
                        (-10, 10), (0.5, 1.0), (0, 2*np.pi),  # amp1, freq1, phase1
                        (-10, 10), (0.5, 1.0), (0, 2*np.pi),  # amp2, freq2, phase2
                        (-10, 10), (0.5, 1.0), (0, 2*np.pi),  # amp3, freq3, phase3
                        (-10, 10)                             # shift
                    ]
                elif  func_data.get('type') == 'sin_sin' or func_data.get('type') == 'cos_cos':
                    bounds = [(-10, 10), (0.1, 0.5), (0, 2*np.pi), (-10, 10),(0.1, 0.5),(0, 2*np.pi),(-10, 10)]
                else:
                    # amp, freq, phase, shift
                    bounds = [(-10, 10), (0.1, 0.5), (0, 2*np.pi), (-10, 10)]
            
            elif classes[j] == 'piecewise':
                bounds = []
                for i, name in enumerate(func.param_names):
                    if 'freq' in name.lower():
                        bounds.append((0.1, 0.5))        # частота freq - важный параметр который влияет на остальные
                    elif 'phase' in name.lower():
                        bounds.append((0, 2*np.pi))      # фаза по кругу
                    else:
                        bounds.append((-10, 10))         # всё остальное не важно

            res_no_del = differential_evolution(
                problem_no_del.total_residual,
                bounds,
                maxiter=200,           
                popsize=15,            
                seed=0,                  
                tol=1e-10,               
                updating='deferred',     
                workers=1,               
                disp=False               
            )
            nfev = res_no_del.nfev
            params_no_del = res_no_del.x
            print(f"  эталонные параметры: {params}")
            print(f"  найденные параметры: {params_no_del}")
            print(f"  ошибка: {np.mean(np.abs(params_no_del - params)):.6f}")
            print(f"  nfev: {nfev}")

            y_no_del = func.evaluate(params_no_del, x_orig)

            with open('Mean_HyperP.pkl', 'rb') as f:
                hyper_params_data = pickle.load(f)
            
            hyper_params = hyper_params_data.get(classes[j])
            problem_del = Model_for_search(
                x=x_orig,
                params=params,
                func=func,
                step= hyper_params.get('step'),
                min_point=hyper_params.get('min_point'),
                rasp='uniform',
                coeff=hyper_params.get('coeff'),
                stable_iterations=hyper_params.get('stable_iterations'),
                delt=1
            )
            problem_del.calc_and_save_func()

            res_del = differential_evolution(
                problem_del.total_residual,
                bounds,
                maxiter=200,           
                popsize=15,            
                seed=0,                  
                tol=1e-10,               
                updating='deferred',     
                workers=1,               
                disp=False               
            )
            nfevd = res_del.nfev
            params_del = res_del.x
            print(f"  найденные c удалением параметры: {params_del}")
            print(f"  ошибка: {np.mean(np.abs(params_del - params)):.6f}")
            print(f"  nfev: {nfevd}")

            x_remaining = problem_del.get_x()
            # y для оставшихся точек
            y_rem = func.evaluate(params_del, x_remaining)
            
            #Восстановленная функция по всем x_orig (с параметрами после удаления)
            y_recovered = func.evaluate(params_del, x_orig)

            create_save_4gr(
                original_data=(x_orig, y_orig),
                no_del_data=(x_orig, y_no_del),
                del_data=(x_remaining, y_rem),
                recovered_data=(x_orig, y_recovered),
                class_name=classes[j],
                func_id=func_data['id'],
                save_dir='Final_Results_Plots'
            )