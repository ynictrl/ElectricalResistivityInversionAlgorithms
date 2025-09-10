import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time

def Ln_u():

  # variaveis iniciais
  x_step = np.log(10) / 3
  x0_main = -8.135341
  x0_inter = x0_main + 0.5 * x_step

  # x=ln(u)
  lnU = np.array([x0_main, x0_inter])
  for i in range(35):
      lnU = np.append(lnU, lnU[i] + x_step)

  return lnU

def res_app(p, h):

  '''
  Return:
  Função que calcula a resistividade aparente

  Input:
  p = resistividade(ohm m) de cada camada | array(n)
  h = espessura(m) de cada camada | array(n-1) | obs: ultima é infinita

  Output:
  Resistidvidade aparente
  '''

  k = np.array([])
  for i in range(3):
      k = np.append(k, (p[i+1]-p[i])/(p[i+1]+p[i]))

  _filter = [0.0284, 0.4582, 1.5662, -1.3341, 0.3473, -0.0935, 0.0416, -0.0253, 0.0179, -0.0067]

  # u
  u = np.array([])
  for i in range(37):
      u = np.append(u, np.exp(Ln_u()[i]))

  # T_AB
  TAB = np.array([])
  for i in range(37):
      TAB = np.append(TAB, p[0]*(1+(-1)*np.exp(-2*h[0]/u[i]))/(1-(-1)*np.exp(-2*h[0]/u[i])))

  # T_BC
  TBC = np.array([])
  for i in range(37):
      TBC = np.append(TBC, p[1]*(1+(-1)*np.exp(-2*h[1]/u[i]))/(1-(-1)*np.exp(-2*h[1]/u[i])))

  # T_CD
  TCD = np.array([])
  for i in range(37):
      TCD = np.append(TCD, p[2]*(1+k[2]*np.exp(-2*h[2]/u[i]))/(1-k[2]*np.exp(-2*h[2]/u[i]))) # ultimo k?

  # T_BCD
  TBCD = np.array([])
  for i in range(37):
      TBCD = np.append(TBCD, (TBC[i]+TCD[i])/(1+TBC[i]*TCD[i]/(p[1]*p[1])))

  # T_ABCD
  TABCD = np.array([])
  for i in range(37):
      TABCD = np.append(TABCD, (TAB[i]+TBCD[i])/(1+TAB[i]*TBCD[i]/p[0]/p[0]))

  # res
  res = np.array([]) #(m)
  for i in range(19):
      res = np.append(res, _filter[0]*TABCD[i+18]+_filter[1]*TABCD[i+16]+_filter[2]*TABCD[i+14]+_filter[3]*TABCD[i+12]+_filter[4]*TABCD[i+10]+_filter[5]*TABCD[i+8]+_filter[6]*TABCD[i+6]+_filter[7]*TABCD[i+4]+_filter[8]*TABCD[i+2]+_filter[9]*TABCD[i])

  return res


# adicionar ruido
def add_gaussian_noise(data, mean=0, std_dev=0.30):
  """
  Adiciona ruído gaussiano a um conjunto de dados.
  Parâmetros:
  - data: np.array, dados originais.
  - mean: média do ruído (padrão: 0).
  - std_dev: desvio padrão do ruído (padrão: 0.01).
  Retorna:
  - np.array com ruído adicionado.
  """
  noise = np.random.normal(mean, std_dev, size=data.shape)
  return data + noise


# Calculo da matriz Jacobiana / matriz sensibilidade
def jacobiana(p, h, d_obs, v = 1e-3):
    """
    Parâmetros:
    - p: resistividade(ohm m) de cada camada | array(n)
    - h: espessura(m) de cada camada | array(n-1) | obs: ultima é infinita
    - d_obs: dado oservado.
    Retorna:
    - matriz Jacobiana.
    """
    n_params = len(p)
    n_dados = len(d_obs)
    G = np.zeros((n_dados, n_params))

    deltas = []
    p_plus_all = []
    p_minus_all = []
    res_plus_all = []
    res_minus_all = []

    for i in range(len(p)):

      delta = p[i] * v
      deltas.append(delta)

      # duvida: quando o p_plus e p_minus muda a cada iteração. Reseta uma nova cópia ou as atualizações permanecem?
      # exemplo: [100, 100, 100]: [101, 100, 100] -> [100, 101, 100] -> [100, 100, 101] ->
      p_plus = p.copy()
      p_minus = p.copy()

      # p_plus[i] = p[i] + delta
      # p_minus[i] = p[i] - delta
      p_plus[i] = p_plus[i] + delta
      p_minus[i] = p_minus[i] - delta

      res_plus = res_app(p_plus, h)
      res_minus = res_app(p_minus, h)

      p_plus_all.append(p_plus)
      p_minus_all.append(p_minus)
      res_plus_all.append(res_plus)
      res_minus_all.append(res_minus)

      G[:,i] = (res_plus - res_minus)/(2*delta)

    # retorna a matriz
    return G, deltas, res_plus_all, res_minus_all, p_plus_all, p_minus_all


def residue(observed_data, predicte_data):
  r = observed_data - predicte_data
  return r


def error(observed_data, predicte_data):
  # phi
  p = (np.linalg.norm(residue(observed_data, predicte_data)))**2
  return p


# def gauss_newton(p_init, _h, d_obs, max_iter, tol_delta=1e-3, tol_phi=1e-3):
def gauss_newton(params):
    """
    Ajustar parâmetros aplicando o método Gauss Newton.

    Parâmetros:
    - params (dict): Dicionário com os seguintes campos:
        - p_init: hipóetses iniciais dos parâmetros | array.
        - _h: espessura(m) de cada camada | array(n-1) | obs: ultima é infinita.
        - d_obs: dados observados | array.

        - i: número de iterações.
        - tol_delta: tolerância para norma da atualização do parâmetro.
        - tol_phi: tolerância para variação da função objetivo (erro).

    Retorna:
    - p_atual: parâmetros ajustados.
    - i: número de iterações.
    - ps: histórico dos parâmetros.
    - phis: histórico dos valores da função objetivo | array.
    """
    d_obs = params['d_obs']
    p_init = params['p_init']
    h = params['h']
    max_iter = params.get('tol_phi', 100)
    # tol_delta = params.get('tol_delta', 1e-3)
    tol_phi = params.get('tol_phi', 1e-3)

    # Início da contagem de tempo
    tempo_inicial = time.time()

    j = 0 # iterações

    p0 = p_init.copy()  # parametros antecessores
    pred0 = res_app(p0, h)  # dado predito antecessor
    r0 = d_obs - pred0  # resíduo antecessor
    phi0 = (np.linalg.norm(d_obs - pred0))**2  # phi antecessor

    p0s = [p0]  # lista de parametros
    phis = [phi0]  # lista de phis

    for j in range(max_iter):

        # Definição da jacobiana e resíduos
        G = jacobiana(p0, h, d_obs)[0]
        r = d_obs - res_app(p0, h)

        GTG = G.T @ G
        GTr = G.T @ r
        GTG += np.eye(len(p_init)) * 1e-8

        # Adicionar valores para os parametros
        delta = np.linalg.solve(GTG,GTr)
        p_atual = p0 + delta

        # Calcular dado predito
        pred = res_app(p_atual, h)

        # Função objetivo (norma² do resíduo)
        # TRANSFORMAR EM FUNÇÃO PHI E RES
        # RES = D_obs - pred
        phi = error(d_obs, pred)
        phis.append(phi)

        # check a convergência
        # se a norma das atualizações dos parâmetros for menor que tol
        if phi < tol_phi:
          print('Convergência por variação dos parâmetros')
          break

        if max_iter > 0 and abs(phi - phis[-2]) < tol_phi:
          print("Convergência por estabilidade de Phi")
          break

        p0 = p_atual.copy()
        p0s.append(p0)

    # duvida: a função retorna um ajuste da res. app. ou da res?
    tempo_final = time.time()
    tempo = tempo_final - tempo_inicial
    return p_atual, j, p0s, phis, tempo


def annealing(params):
    """
    Ajustar parâmetros com o método Annealing.

    Parâmetros:
    - params (dict): Dicionário com os seguintes campos:
        - F: Função.
        - d_obs: Dado observado.
        - p_init: Parâmetros iniciais.
        - h: espessura(m) de cada camada (ultima é infinita).
        - t: "Temperatura inicial".
        - r: "Fator de resfriamento".
        - k: "Número de repetições por temperatura".
        - i_max: Máximo de iterações.
        - tol_phi: Tolerância do erro.
        - range_ann: "Intervalo do passo anneling".

    Retorna:
    - p_atual: parâmetros ajustados
    - i+1: iterações
    - ps: lista de paramêtros
    - phis: lista de erro
    - delta_phis: lista da variação do erro
    """
    d_obs = params['d_obs']
    p_init = params['p_init']
    h = params['h']
    t = params.get('t', 5)
    r = params.get('r', 0.8)
    k = params.get('k', 5)
    F = params.get('F', res_app)
    i_max = params.get('i_max', 1000)
    tol_phi = params.get('tol_phi', 1e-3)
    range_ann = params.get('range_ann', 1)

    # Início da contagem de tempo
    tempo_inicial = time.time()

    # por clip do range
    n_params = len(p_init)
    p_atual = p_init.copy() #paramêtros
    pred = F(p_init, h) #dado predito
    phi_atual = error(d_obs, pred) #função objetivo ou erro

    ps = [p_atual] #lista de paramêtros
    phis = [phi_atual] #lista de erro
    delta_phis = [] #lista da variação do erro

    num_solucoes = 10
    p_novo = np.zeros(4)

    for i in range(i_max):

      for _ in range(k):
        # gera r vizinhos aleatórios para os parametrosa a e b

        # passo do anneling
        for j in range(n_params):
          p_novo[j] = p_atual[j] + np.random.uniform(-range_ann, range_ann) # Criar função que sorteia, mudar os valores de acordo com o antigo
        # print(p_novo)
          # p_novo[1] = p_atual[1] + np.random.uniform(0, 1) # o range precisa ser um parametro do anneling

        #calculo do novo erro
        pred_novo = F(p_novo, h)
        phi_novo = error(d_obs, pred_novo)

        delta_phi = phi_novo - phi_atual
        delta_phis.append(delta_phi)

        #critério de SA
        #   prob > teste
        if delta_phi <= 0 or np.random.rand() < np.exp(-delta_phi / t):
          p_atual = p_novo.copy()
          # p_atual[1] = p_novo[1] # .copy() apenas para arrray
          phi_atual = phi_novo.copy()
          ps.append(p_atual)
          phis.append(phi_atual)

        #critério de parada por convergência
        if abs(delta_phi) < tol_phi:
          print('Critério de parada: delta_phi < tol_phi')
          # break!1
          tempo_final = time.time()
          tempo = tempo_final - tempo_inicial
          return p_atual, i+1, ps, phis, delta_phis, tempo

      # Resfriamento
      t *= r

    tempo_final = time.time()
    tempo = tempo_final - tempo_inicial
    print('Critério de parada: iteração máxima')
    return p_atual, i+1, ps, phis, delta_phis, tempo


# TESTE


p_true = np.array([450, 100, 400, 800])
h = np.array([5, 7, 5, 10])

d_obs = res_app(p_true, h)
d_obs_n = add_gaussian_noise(d_obs)

p_init = np.array([10, 10, 10, 10])
# p_init = np.array([100, 100, 100, 100])


# parametros_ann = {
#     'd_obs': d_obs,
#     'p_init': p_init,
#     'h': h,
# }

# p_ann, j_ann, ps_ann, phis_ann, dphis_ann, t_ann = annealing(parametros_ann)

# print("Resultado:")
# print("p_ajusted =", p_ann)
# print("Erro final =", phis_ann[-1])
# print(f'Iterações: {j_ann}')
# print("tempo_final", t_ann)

# print("------------------------------------------")

parametros_gn = {
    'd_obs': d_obs,
    'p_init': p_init,
    'h': h,
}

p_gn, j_gn, p0s_gn, phis_gn, t_gn = gauss_newton(parametros_gn)

print("Resultado:")
print("p_ajusted =", p_gn)
print("Erro final =", phis_gn[-1])
print(f'Iterações: {j_gn}')
print("tempo_final", t_gn)

