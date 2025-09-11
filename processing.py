import nick_inversion as nick
import numpy as np
import matplotlib.pyplot as plt
import random

p_init = np.array([100, 100, 100, 100]) # Valores das resistividades (dado inicial)
p_true = np.array([450, 100, 400, 800]) # Valores das resistividades (dado verdadeiro)
h = np.array([5, 7, 5, 10]) # Comprimento das camadas

d_obs = nick.res_app(p_true, h) # Dado observado (resistividade aparente)
d_obs_n = nick.add_gaussian_noise(d_obs, 0, 0.5) # Dado observado com ruído

# parametros do annealing
parametros_ann = {
    'd_obs': d_obs_n,
    'p_init': p_init,
    'h': h
}

p_ann, j_ann, ps_ann, phis_ann, dphis_ann, t_ann = nick.annealing(parametros_ann)

print("Resultado Annealing:")
print(f'p_ajusted = {p_ann}')
print(f'Erro final = {phis_ann[-1]}')
print(f'Iterações: {j_ann}')
print(f'tempo_final {t_ann}')

print("------------------------------------------")

# parametros do gauss newton
parametros_gn = {
    'd_obs': d_obs_n,
    'p_init': p_init,
    'h': h
}

p_gn, j_gn, p0s_gn, phis_gn, t_gn =  nick.gauss_newton(parametros_gn)

print("Resultado Gauss Newton:")
print(f'p_ajusted = {p_gn}')
print(f'Erro final = {phis_gn[-1]}')
print(f'Iterações: {j_gn}')
print(f'tempo_final {t_gn}')