import numpy as np
import matplotlib.pyplot as plt

# Carica il file .npz
# Assumendo che opt.results e basename siano definiti correttamente
npz_file = np.load('/Users/kolyszko/PycharmProjects/SuperResolution/results/image_111_lr2.npz')

# Supponiamo che l'array di interesse sia 'out'
# Estrai la prima banda;
# il modo di indicizzare dipende dalla struttura dell'array:
# ad esempio, se l'array ha forma (num_bande, altezza, larghezza)
first_band = npz_file['out'][0]

# Ciclo su ogni banda
for i in range(first_band.shape[0]):
    # Estrai la banda e trasponila se necessario (ad esempio, da (480,240) a (240,480))
    band = first_band[i].T

    # Crea una nuova figura per ogni banda
    plt.figure()
    plt.imshow(band, cmap='gray')
    plt.title("Banda {}".format(i))
    plt.colorbar()
    plt.show()
