import nick_inversion as nick
import numpy as np

def ModelGenerator(min_p, max_p, N, qnt_p=4):
    # Gerador de vetores de modelos

    model_matriz = np.zeros((N, qnt_p))
    for i in range(N):
        model_matriz[i] = np.random.randint(min_p, max_p, qnt_p)

    return model_matriz

def ResAppGenerator(h, m_models):
    # Gerador de vetores de resistividade aparente

    N = np.shape(m_models)[0]
    M = 19
    data_matriz = np.zeros((N,M))

    for i in range(N):
        p_gen = nick.res_app(m_models[i], h)

        data_matriz[i] = p_gen

    return data_matriz
    