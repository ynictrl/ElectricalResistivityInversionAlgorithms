import nick_inversion as nick
import numpy as np
import matplotlib.pyplot as plt
import random

d_obs = np.array([198, 160, 140, 112, 95, 84, 79, 84, 82, 92, 101, 100, 102])

h = np.array([0.7, 2.1, 3, 100]) # Comprimento das camadas

p_init = np.array([150, 50, 50, 80]) # Valores das resistividades (dado inicial)

# parametros do annealing
parametros_ann = {
    'd_obs': d_obs,
    'p_init': p_init,
    'h': h,
    'n_A': 13
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
    'd_obs': d_obs,
    'p_init': p_init,
    'h': h,
    'n_A': 13
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
DATA_A = np.array([0.47, 0.69, 1, 1.47, 2.15, 3.16, 4.64, 6.81, 10, 14.68, 21.54, 31.62, 46.42])

# dado predito (Annealing)
pred_resapp_ann = nick.res_app(p_ann, h, 13)

# dado predito (Gauss Newton)
pred_resapp_gn = nick.res_app(p_gn, h, 13)

# p_true = nick.res_app(d_obs, h, 13)

# Plot
plt.figure(figsize=(8,5))
plt.plot(DATA_A, d_obs, 'r.-', label='Observed data')
plt.plot(DATA_A, pred_resapp_ann, 'b.-', label='Annealing data')
plt.plot(DATA_A, pred_resapp_gn, 'g.-', label='Gauss Newton data')
plt.title('Data Inversion - SEV')
plt.xlabel('Electrode separation (a-spacing) in m')
plt.ylabel('Apparent resistivity in ohm m')
plt.xscale('log')
# plt.yscale('log')
# plt.barplot()
# plt.yticks([100, 200, 400, 600, 800])
plt.tight_layout()
plt.legend()
plt.grid(True)
plt.show()