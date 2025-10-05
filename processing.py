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


# ------------------------------------------------------------------ #

# transformar em um arquivo
# espaçamento dos eletrodos
DATA_A = [0.099998630799061, 0.146777917050071, 0.215440519149188,
          0.316223436223296, 0.464152528093489, 0.681282740800554,
          0.99998630799061, 1.46777917050071, 2.15440519149188,
          3.16223436223296, 4.64152528093489, 6.81282740800554,
          9.99986307990611, 14.6777917050072, 21.5440519149188,
          31.6223436223296, 46.4152528093489, 68.1282740800555, 99.9986307990611]

# dado predito (Annealing)
pred_resapp_ann = nick.res_app(p_ann, h)

# dado predito (Gauss Newton)
pred_resapp_gn = nick.res_app(p_gn, h)


plt.figure(figsize=(8,5))
plt.plot(DATA_A, d_obs_n, 'ro', label='Observed data')
plt.plot(DATA_A, pred_resapp_ann, 'b.-', label='Annealing data')
plt.plot(DATA_A, pred_resapp_gn, 'g.-', label='Gauss Newton data')
plt.title('Data Inversion - SEV')
plt.xlabel('Electrode separation (a-spacing) in m')
plt.ylabel('Apparent resistivity in ohm m')
plt.xscale('log')
# plt.yscale('log')
# plt.barplot()
plt.yticks([100, 200, 400, 600, 800])
plt.tight_layout()
plt.legend()
plt.grid(True)
plt.show()