import nick_inversion as nick
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec as gridspec

p_init = np.array([100, 100, 100, 100]) # Valores das resistividades (dado inicial)
p_true = np.array([450, 100, 400, 800]) # Valores das resistividades (dado verdadeiro)
h = np.array([5, 7, 5, 10]) # Comprimento das camadas

p_ann = np.array([100, 50, 20, 500])
p_gn = np.array([90, 60, 25, 450])
p_true = np.array([95, 55, 22, 480])
# Plot

M = len(p_ann) # Número de parâmetros
z0 = 0  # Profundidade do topo da primeira camada
h = np.array([5, 7, 5, 10]) # Comprimento das camadas

interfaces = np.zeros(M)
for l in range(M):
    interfaces[l] = z0 + sum(h[:l])

#Adicionar mais um elemneto nos vetores
interfaces_ext = np.append(interfaces, interfaces[-1] + h[-1])
p_ann_ext = np.append(p_ann, p_ann[-1])
p_gn_ext = np.append(p_gn, p_gn[-1])
p_true_ext = np.append(p_true, p_true[-1])

fig = plt.figure(figsize=(15,15))
gs2 = gridspec.GridSpec(3, 3, width_ratios=[2, 4, 4])
gs2.update(left=0.4, right=1.4, hspace=1)

ax1 = plt.subplot(gs2[:, :-2])
ax1.step(p_ann_ext, interfaces_ext, 'r', alpha = 0.75, label = "annealing")
ax1.step(p_gn_ext, interfaces_ext, 'g', alpha = 0.75, label = "gauss newton")
ax1.step(p_true_ext, interfaces_ext, '--k', alpha = 0.75, label = "true")

# fill_betweenx(np.arange(0, self.interfaces[i]), 0, 1, color=self.lithology_respective_color[layer], label=layer)
# ax1.fill_betweenx(interfaces, p_ann, where=(p_ann > p_true), interpolate=True)
# ax1.fill_between(p_gn, interfaces)
# ax1.fill_between(p_true, interfaces)

ax1.set_xlabel('Resistivity (Ohm.m)', fontsize='x-large')
ax1.set_ylabel('Depth (m)', fontsize='x-large')
ax1.set_ylim((interfaces_ext[-1] + 1, z0))  # Invertido: maior profundidade embaixo
ax1.set_title('1-D Resistivity Model', fontsize='x-large')
ax1.tick_params(labelsize=14)
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.4)

plt.show()