from matplotlib import pyplot as plt
import numpy as np
import arviz
from reggae.data_loaders import scaled_barenco_data

def plot_kinetics(labels, k, plot_barenco=False, true_k=None, num_avg=50):
    k_latest = np.mean(k[-num_avg:], axis=0)
    num_genes = k.shape[1]

    plt.figure()
    plt.suptitle('Transcription ODE Kinetic Parameters')
    A = k_latest[:, 0]
    plt.title('Initial Conditions')
    for j in range(num_genes):
        plt.bar(np.arange(num_genes), A, width=0.4, tick_label=labels, label='Model')

    plt.figure(figsize=(14, 14))
    B = k_latest[:,1]
    D = k_latest[:,2]
    S = k_latest[:,3]
    data = [B, S, D]
    true_data = [None, None, None]

    comparison_label = 'True'
    if plot_barenco:
        comparison_label = 'Barenco et al.'
        # From Martino paper ... do a rough rescaling so that the scales match.
        B_barenco = np.array([2.6, 1.5, 0.5, 0.2, 1.35])[[0, 4, 2, 3, 1]]
        B_barenco = B_barenco/np.mean(B_barenco)*np.mean(B)
        S_barenco = (np.array([3, 0.8, 0.7, 1.8, 0.7])/1.8)[[0, 4, 2, 3, 1]]
        S_barenco = S_barenco/np.mean(S_barenco)*np.mean(S)
        D_barenco = (np.array([1.2, 1.6, 1.75, 3.2, 2.3])*0.8/3.2)[[0, 4, 2, 3, 1]]
        D_barenco = D_barenco/np.mean(D_barenco)*np.mean(D)
        true_data = [B_barenco, S_barenco, D_barenco]
    elif true_k is not None:
        true_data = true_k
    plot_labels = ['Basal rates', 'Sensitivities', 'Decay rates']

    plotnum = 331
    for A, B, label in zip(data, true_data, plot_labels):
        plt.subplot(plotnum)
        plotnum+=1
        plt.bar(np.arange(num_genes)-0.2, A, width=0.4, tick_label=labels, label='Model')
        if B is not None:
            plt.bar(np.arange(num_genes)+0.2, B, width=0.4, color='blue', align='center', label=comparison_label)
        plt.title(label)
        plt.legend()


def plot_kinetics_convergence(k, δ):
    num_genes = k.shape[1]
    labels = ['a', 'b', 'd', 's']
    plt.figure(figsize=(14, 14))
    plt.suptitle('Convergence of ODE Kinetic Parameters')
    for j in range(num_genes):
        ax = plt.subplot(num_genes, 2, j+1)
        k_param = k[:, j, :]
        
        for i in range(4):
            plt.plot(k_param[:, i], label=labels[i])
        plt.axhline(np.mean(k_param[-50:, 3]))
        plt.legend()
        ax.set_title(f'Gene {j}')
    plt.tight_layout()

    plt.figure(figsize=(6, 4))
    for i in range(δ.shape[1]):
        plt.plot(δ[:, i], label=f'TF {i}')
    plt.title('δ Convergence')


def plot_genes(titles, m_preds, data, num_hpd=20):
    num_hpd = max(num_hpd, 20)
    plt.figure(figsize=(14, 17))
    plt.suptitle('Genes')
    num_genes = m_preds[0].shape[0]
    N_p = m_preds[0].shape[1]
    for j in range(num_genes):
        ax = plt.subplot(531+j)
        plt.title(titles[j])
        plt.scatter(data.common_indices, data.m_obs[j], marker='x', label='Observed')
        # plt.errorbar([n*10+n for n in range(7)], Y[j], 2*np.sqrt(Y_var[j]), fmt='none', capsize=5)

        for i in range(1, 20):
            plt.plot(m_preds[-i][j,:], color='grey', alpha=0.5)
            
        # HPD:
        bounds = arviz.hpd(m_preds[-num_hpd:, j,:], credible_interval=0.95)
        plt.fill_between(np.arange(N_p), bounds[:, 0], bounds[:, 1], color='grey', alpha=0.3, label='95% credibility interval')

        plt.xticks(np.arange(N_p)[data.common_indices.numpy()])
        ax.set_xticklabels(np.arange(data.t[-1]))
        plt.xlabel('Time (h)')
        plt.legend()
    plt.tight_layout()

def plot_tf(data, f_samples, num_hpd=20, plot_barenco=True):
    t = data.t
    τ = data.τ
    common_indices = data.common_indices.numpy()
    num_tfs = data.f_obs.shape[0]
    fig = plt.figure(figsize=(13, 7*np.ceil(num_tfs/2)))
    plt.suptitle('Transcription Factors')

    # if 'σ2_f' in model.params._fields:
    #     σ2_f = model.params.σ2_f.value
    #     plt.errorbar(τ[common_indices], f_observed[0], 2*np.sqrt(σ2_f[0]), 
    #                 fmt='none', capsize=5, color='blue')
    # else:
    #     σ2_f = σ2_f_pre
    horizontal_subplots = 21 if num_tfs > 1 else 11
    for i in range(num_tfs):
        
        plt.subplot(num_tfs*100+horizontal_subplots+i)
        for s in range(1,20):
            f_i = f_samples[-s]

            kwargs = {}
            if s == 1:
                kwargs = {'label':'Samples'}
            plt.plot(τ, f_i[i], c='cadetblue', alpha=0.5, **kwargs)


        plt.scatter(τ[common_indices], data.f_obs[i], marker='x', s=70, linewidth=3, label='Observed')

        # HPD:
        bounds = arviz.hpd(f_samples[-num_hpd:,i,:], credible_interval=0.95)
        plt.fill_between(τ, bounds[:, 0], bounds[:, 1], color='grey', alpha=0.3, label='95% credibility interval')
        plt.xticks(t)
        fig.axes[0].set_xticklabels(t)
        plt.xlabel('Time (h)')
        plt.legend()

        if plot_barenco:
            barenco_f, _ = scaled_barenco_data(np.mean(f_samples[-10:], axis=0))
            plt.scatter(τ[common_indices], barenco_f, marker='x', s=60, linewidth=3, label='Barenco et al.')

        plt.scatter(τ[common_indices], data.f_obs[i], marker='x', s=70, linewidth=3, label='Observed')

        plt.fill_between(τ, bounds[:, 0], bounds[:, 1], color='grey', alpha=0.3, label='95% credibility interval')
        plt.xticks(t)
        fig.axes[0].set_xticklabels(t)
        plt.xlabel('Time (h)')
        plt.legend()
    plt.tight_layout()

def generate_report(data, 
                    k_samples, 
                    δ_samples, 
                    f_samples, 
                    σ2_m_samples, 
                    rbf_param_samples,
                    m_preds,
                    gene_names=None,
                    num_avg=20,
                    plot_barenco=True,
                    true_k=None,
                    num_hpd=20):

    if gene_names is None:
        gene_names = np.arange(data.m_obs.shape[0])
    plot_tf(data, f_samples, plot_barenco=plot_barenco)

    plot_genes(gene_names, m_preds, data, num_hpd=num_hpd)

    plot_kinetics_convergence(k_samples, δ_samples)

    plot_kinetics(gene_names, k_samples, true_k=true_k,
                  num_avg=num_avg, plot_barenco=plot_barenco)
                  
    plt.figure(figsize=(10, 4))
    plotnum = 0
    for name, param in (zip(['L', 'V'], rbf_param_samples)):
        ax = plt.subplot(221+plotnum)
        plt.plot(param)
        ax.set_title(name)
        plotnum+=1

