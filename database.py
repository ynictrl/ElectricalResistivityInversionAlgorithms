import nick_inversion as nick
import numpy as np

def ResAppGenerator():
    # Gerador de vetores de resistividade aparente

    # elementos do vetor de resistividade
    a = np.random.uniform(50, 900, 2) 
    b = np.random.uniform(50, 900, 2)
    c = np.random.uniform(50, 900, 2)
    d = np.random.uniform(50, 900, 2)

    # vetor de camadas
    h = np.array([5, 7, 5, 10]) # Comprimento das camadas

    N = 19
    M = len(a) * len(b) * len(c) * len(d)
    matriz_p = np.zeros((M,N))
    i_matriz = 0
        
    for i in a:
        for j in b:
            for k in c:
                for l in d:
                    p_init_gen = np.array([i, j, k, l]) 
                    p_gen = nick.res_app(p_init_gen, h)

                    matriz_p[i_matriz] = p_gen
                    i_matriz += 1
                    
    return matriz_p

rag = ResAppGenerator()
print(rag, len(rag))
# print(a,b,c,d)
# print(nick.res_app(np.array([a[0],b[0],c[0],d[0]]), h))
    