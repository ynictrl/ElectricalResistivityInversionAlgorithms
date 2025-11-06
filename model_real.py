import nick_inversion as nick
import processing19 as pro
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
# p_true_ext = np.append(pro.p_true, pro.p_true[-1])

fig = plt.figure(figsize=(12,8))
gs2 = gridspec.GridSpec(3, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1, 1])
gs2.update(hspace=0.8, wspace=0.5)
#gs2.update(left=0.1, right=0.9, wspace=0.5, hspace=1)

#---------------------------------------------------------------------------------------

# Plot da diferença dos métodos

ax1 = plt.subplot(gs2[:, 0])
# ax1.step(p_true_ext, interfaces_ext, '--r', alpha = 0.75, label = "Observed data")
ax1.step(p_ann_ext, interfaces_ext, 'b', alpha = 0.75, label = "Annealing data")
ax1.step(p_gn_ext, interfaces_ext, 'g', alpha = 0.75, label = "Gauss Newton data")
ax1.set_xlabel('Resistivity (Ohm.m)', fontsize='x-large')
ax1.set_ylabel('Depth (m)', fontsize='x-large')
ax1.set_ylim((interfaces_ext[-1] + 1, z0))  # Invertido: maior profundidade embaixo
ax1.set_title('1-D Resistivity Model', fontsize='x-large')
ax1.tick_params(labelsize=14)
plt.grid()
plt.legend()
plt.yticks(interfaces_ext)

#---------------------------------------------------------------------------------------

# Plot do anneling

ax2 = plt.subplot(gs2[:, 1]) # ou plt.subplot(gs2[:, :-2])

N_color = 50
cmap = plt.get_cmap('coolwarm', N_color)
colors = [cmap(i) for i in range(N_color)]

def choseColor(vet, inter):
  n1 = 100
  n2 = 1000 # n2>=600

  rangeN = np.linspace(n1, n2, N_color)
  dif = rangeN[1]-rangeN[0]

  for j in range(len(rangeN)):
    if vet[inter] >= (rangeN[j]-dif) and vet[inter] <= rangeN[j]:
      print(j)
      return colors[j]

for i in range(M):
    top = interfaces_ext[i]
    bottom = interfaces_ext[i + 1]

    ax2.fill_betweenx(
      y=[top, bottom],
      x1=0,
      x2=max(p_gn_ext.max(), p_ann_ext.max()) * 1.1,
      color=choseColor(p_ann_ext, i),
      alpha=0.7,
      # edgecolor='white'
    )

ax2.step(p_ann_ext, interfaces_ext, 'b', alpha = 0.75, label = "Annealing data")
ax2.set_xlabel('Resistivity (Ohm.m)', fontsize='x-large')
ax2.set_ylabel('Depth (m)', fontsize='x-large')
ax2.set_ylim((interfaces_ext[-1] + 1, z0))  # Invertido: maior profundidade embaixo
ax2.set_title('Annealing Model', fontsize='x-large')
ax2.tick_params(labelsize=14)
plt.yticks(interfaces_ext)

#-----------------------------------------------------------------------------------

# Plot do gauss newton

ax3 = plt.subplot(gs2[:, 2]) # ou plt.subplot(gs2[:, :-2])

N_color = 50
cmap = plt.get_cmap('coolwarm', N_color)
colors = [cmap(i) for i in range(N_color)]

def choseColor(vet, inter):
  n1 = 100
  n2 = 1000 # n2>=600

  rangeN = np.linspace(n1, n2, N_color)
  dif = rangeN[1]-rangeN[0]

  for j in range(len(rangeN)):
    if vet[inter] >= (rangeN[j]-dif) and vet[inter] <= rangeN[j]:
      print(j)
      return colors[j]

for i in range(M):
    top = interfaces_ext[i]
    bottom = interfaces_ext[i + 1]

    ax3.fill_betweenx(
      y=[top, bottom],
      x1=0,
      x2=max(p_gn_ext.max(), p_ann_ext.max()) * 1.1,
      color=choseColor(p_gn_ext, i),
      alpha=0.7,
      # edgecolor='white'
    )

ax3.step(p_gn_ext, interfaces_ext, 'g', alpha = 0.75, label = "Gauss Newton data")
ax3.set_xlabel('Resistivity (Ohm.m)', fontsize='x-large')
ax3.set_ylabel('Depth (m)', fontsize='x-large')
ax3.set_ylim((interfaces_ext[-1] + 1, z0))  # Invertido: maior profundidade embaixo
ax3.set_title('Gauss Newton Model', fontsize='x-large')
ax3.tick_params(labelsize=14)
plt.yticks(interfaces_ext)

plt.tight_layout()
plt.show()