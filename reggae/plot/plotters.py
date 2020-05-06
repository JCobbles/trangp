from matplotlib import pyplot as plt
import numpy as np

def plot_kinetics(m_df, kbar, plot_barenco=False):
    plt.figure(figsize=(14, 14))
    num_genes = kbar.shape[1]
    k_latest = np.exp(np.mean(kbar[-100:], axis=0))
    B = k_latest[:,1]
    D = k_latest[:,2]
    S = k_latest[:,3]
    data = [B, S, D]
    barenco_data = [None, None, None]

    if plot_barenco:
        # From Martino paper ... do a rough rescaling so that the scales match.
        B_barenco = np.array([2.6, 1.5, 0.5, 0.2, 1.35])[[0, 4, 2, 3, 1]]
        B_barenco = B_barenco/np.mean(B_barenco)*np.mean(B)
        S_barenco = (np.array([3, 0.8, 0.7, 1.8, 0.7])/1.8)[[0, 4, 2, 3, 1]]
        S_barenco = S_barenco/np.mean(S_barenco)*np.mean(S)
        D_barenco = (np.array([1.2, 1.6, 1.75, 3.2, 2.3])*0.8/3.2)[[0, 4, 2, 3, 1]]
        D_barenco = D_barenco/np.mean(D_barenco)*np.mean(D)
        barenco_data = [B_barenco, S_barenco, D_barenco]

    labels = ['Basal rates', 'Sensitivities', 'Decay rates']

    plotnum = 331
    for A, B, label in zip(data, barenco_data, labels):
        plt.subplot(plotnum)
        plotnum+=1
        plt.bar(np.arange(num_genes)-0.2, A, width=0.4, tick_label=m_df.index, label='Model')
        if B is not None:
            plt.bar(np.arange(num_genes)+0.2, B, width=0.4, color='blue', align='center', label='Barenco et al.')
        plt.title(label)
        plt.legend()


def plot_kinetics_convergence(kbar):
    num_genes = kbar.shape[1]
    labels = ['a', 'b', 'd', 's']
    plt.figure(figsize=(14, 14))
    plt.title('Transcription ODE kinetic parameters')
    for j in range(num_genes):
        ax = plt.subplot(num_genes, 2, j+1)
        k_param = kbar[:, j, :]
        
        for k in range(4):
            plt.plot(k_param[:, k], label=labels[k])
        plt.axhline(np.mean(k_param[-200:, 3]))
        plt.legend()
        ax.set_title(f'Gene {j}')
    plt.tight_layout()
