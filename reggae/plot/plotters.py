from matplotlib import pyplot as plt
import numpy as np
import arviz

def plot_kinetics(m_df, k, plot_barenco=False, num_avg=50):
    k_latest = np.mean(k[-num_avg:], axis=0)
    num_genes = k.shape[1]
    plt.figure()
    A = k_latest[:, 0]
    for j in range(num_genes):
        plt.bar(np.arange(num_genes), A, width=0.4, tick_label=m_df.index, label='Model')

    plt.figure(figsize=(14, 14))
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


def plot_kinetics_convergence(k):
    num_genes = k.shape[1]
    labels = ['a', 'b', 'd', 's']
    plt.figure(figsize=(14, 14))
    plt.title('Transcription ODE kinetic parameters')
    for j in range(num_genes):
        ax = plt.subplot(num_genes, 2, j+1)
        k_param = k[:, j, :]
        
        for i in range(4):
            plt.plot(k_param[:, i], label=labels[i])
        plt.axhline(np.mean(k_param[-50:, 3]))
        plt.legend()
        ax.set_title(f'Gene {j}')
    plt.tight_layout()


def plot_genes(m_df, m_preds, data, num_hpd=20):
    num_genes = m_preds[0].shape[0]
    N_p = m_preds[0].shape[1]
    for j in range(num_genes):
        ax = plt.subplot(531+j)
        plt.title(m_df.index[j])
        plt.scatter(data.common_indices, data.m_obs[j], marker='x', label='Observed')
        # plt.errorbar([n*10+n for n in range(7)], Y[j], 2*np.sqrt(Y_var[j]), fmt='none', capsize=5)

        for i in range(1, num_hpd):
            plt.plot(m_preds[-i][j,:], color='grey', alpha=0.5)
            
        # HPD:
        bounds = arviz.hpd(m_preds[:, j,:], credible_interval=0.95)
        plt.fill_between(np.arange(N_p), bounds[:, 0], bounds[:, 1], color='grey', alpha=0.3, label='95% credibility interval')

        plt.xticks(np.arange(N_p)[data.common_indices])
        ax.set_xticklabels(np.arange(data.t[-1]))
        plt.xlabel('Time (h)')
        plt.legend()
