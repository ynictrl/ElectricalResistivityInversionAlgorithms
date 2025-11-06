import nick_inversion as nick
import processing as pro
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec as gridspec

# Plot

M = len(pro.p_ann) # Número de parâmetros
z0 = 0  # Profundidade do topo da primeira camada

interfaces = np.zeros(M)
for l in range(M):
    interfaces[l] = z0 + sum(pro.h[:l])

#Adicionar mais um elemneto nos vetores
interfaces_ext = np.append(interfaces, interfaces[-1] + pro.h[-1])
p_ann_ext = np.append(pro.p_ann, pro.p_ann[-1])
p_gn_ext = np.append(pro.p_gn, pro.p_gn[-1])
p_true_ext = np.append(pro.p_true, pro.p_true[-1])

fig = plt.figure(figsize=(14,10))
gs2 = gridspec.GridSpec(3, 3, width_ratios=[1, 3, 0], height_ratios=[1, 0.6, 1])
gs2.update(hspace=0.8, wspace=0.5)
#gs2.update(left=0.1, right=0.9, wspace=0.5, hspace=1)

#---------------------------------------------------------------------------------------

# Plot do gráfico estatigrafico

ax1 = plt.subplot(gs2[:, :-2])

N_color = 50
cmap = plt.get_cmap('coolwarm', N_color)
colors = [cmap(i) for i in range(N_color)]

def choseColor(vet, inter):
  # escolhe cores baseando no valor de res
  n1 = 100
  n2 = 1000 # n2>=600
  
  rangeN = np.linspace(n1, n2, N_color)
  dif = rangeN[1]-rangeN[0]

  for j in range(len(rangeN)):
    if vet[inter] >= (rangeN[j]-dif) and vet[inter] <= rangeN[j]:
      # #print(j)
      return colors[j]

for i in range(M):
    # adicionar cores nas camadas
    top = interfaces_ext[i]
    bottom = interfaces_ext[i + 1]

    ax1.fill_betweenx(
      y=[top, bottom],
      x1=0,
      x2=max(p_ann_ext.max(), p_gn_ext.max(), p_true_ext.max()) * 1.1,
      color=choseColor(p_true_ext, i),
      alpha=0.8,
      # edgecolor='white'
    )

ax1.step(p_true_ext, interfaces_ext, '--r', alpha = 0.8, label = "Observed data")
ax1.step(p_ann_ext, interfaces_ext, 'b', alpha = 0.8, label = "Annealing data")
ax1.step(p_gn_ext, interfaces_ext, 'g', alpha = 0.8, label = "Gauss Newton data")
ax1.set_xlabel('Resistivity (Ohm.m)', fontsize='x-large')
ax1.set_ylabel('Depth (m)', fontsize='x-large')
ax1.set_ylim((interfaces_ext[-1] + 1, z0))  # Invertido: maior profundidade embaixo
ax1.set_title('1-D Resistivity Model', fontsize='x-large')
ax1.tick_params(labelsize=14)
plt.yticks(interfaces_ext)
plt.legend()
# plt.grid(True, linestyle='--', alpha=0.4)

#---------------------------------------------------------------------------------------

# Plot da inversão

ax2 = plt.subplot(gs2[:-1, -2])
ax2.plot(pro.DATA_A, pro.d_obs_n, 'r.-', label='Observed data')
ax2.plot(pro.DATA_A, pro.pred_resapp_ann, 'b.-', label='Annealing data')
ax2.plot(pro.DATA_A, pro.pred_resapp_gn, 'g.-', label='Gauss Newton data')
plt.legend(fontsize='x-large',numpoints = 1)
plt.title('Data Inversion - SEV', fontsize='x-large')
plt.xlabel('Electrode separation (a-spacing) in m', fontsize='x-large')
plt.ylabel('Apparent resistivity in ohm m', fontsize='x-large')
plt.xscale('log')
# plt.yscale('log')
plt.yticks([100, 200, 400, 600, 800, 1000])
# plt.tight_layout()
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)


#---------------------------------------------------------------------------------------

# Plot da convergencia 

ax3_base = plt.subplot(gs2[-1, -2])
ax3_base.plot(np.arange(len(pro.phis_ann)), pro.phis_ann, 'b-', label='Annealing data')
ax3_base.set_xlabel('Convergence ANN', fontsize='x-large')
ax3_base.set_ylabel('phi', fontsize='x-large')

ax3_top = ax3_base.twiny()  # cria um segundo eixo X independente (no topo)
ax3_top.plot(np.arange(len(pro.phis_gn)), pro.phis_gn, 'g-', label='Gauss Newton data')

ax3_top.set_xlabel('Convergence GN', fontsize='x-large')

lns1, labs1 = ax3_base.get_legend_handles_labels()
lns2, labs2 = ax3_top.get_legend_handles_labels()
ax3_base.legend(lns1 + lns2, labs1 + labs2, loc='best', fontsize='large')

ax3_base.tick_params(axis='x', colors='blue')
ax3_top.tick_params(axis='x', colors='green')

ax3_base.xaxis.label.set_color('blue')
ax3_top.xaxis.label.set_color('green')

ax3_top.spines['bottom'].set_color('blue')
ax3_top.spines['top'].set_color('green')

plt.tight_layout()
plt.show()