import nick_inversion as nick
import numpy as np

p_init = np.array([100, 100, 100, 100]) # Valores das resistividades (dado inicial)
p_true = np.array([450, 100, 400, 800]) # Valores das resistividades (dado verdadeiro)
h = np.array([5, 7, 5, 10]) # Comprimento das camadas

d_obs = nick.res_app(p_true, h) # Dado observado (resistividade aparente)
d_obs_n = nick.add_gaussian_noise(d_obs, 0, 0.5) # Dado observado com ruído

# separar em grupos(agrupamento de dados) (crossplot)

a = np.linspace(440, 460, 2)
b = np.linspace(90, 110, 2)
c = np.linspace(390, 410, 2)
d = np.linspace(790, 810, 2)

def dataGenerator():
    ...
    # range de (90,800)
    # [init, final, skip]

    N = 4
    M = len(a) * len(b) * len(c) * len(d)
    matriz_p_ann = np.zeros((M,N))
    i_matriz = 0
        
    for i in a:
        for j in b:
            for k in c:
                for l in d:
                    p_init_gen = np.array([i, j, k, l]) 
                    parametros_ann_gen = {
                        'd_obs': d_obs_n,
                        'p_init': p_init_gen,
                        'h': h
                    }
                    p_ann_gen = nick.annealing(parametros_ann_gen)[0]

                    # matriz_p_ann = np.append(matriz_p_ann, p_ann_gen)
                    matriz_p_ann[i_matriz] = p_ann_gen

                    i_matriz += 1
                    


    
    return matriz_p_ann

print(dataGenerator())
    