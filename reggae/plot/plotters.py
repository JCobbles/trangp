from matplotlib import pyplot as plt
import numpy as np
import arviz
from reggae.data_loaders import scaled_barenco_data
from reggae.models.results import SampleResults
from dataclasses import dataclass

@dataclass
class PlotOptions:
    num_plot_genes: int = 20
    num_plot_tfs:   int = 20
    gene_names:     list = None
    tf_names:       list = None
    plot_barenco:   bool = False
    num_kinetic_avg:int = 50
    num_hpd:        int = 100

class Plotter():
    def __init__(self, data, options: PlotOptions):
        self.opt = options
        if self.opt.gene_names is None:
            self.opt.gene_names = np.arange(data.m_obs.shape[1])
        if self.opt.tf_names is None:
            self.opt.tf_names = [f'TF {i}' for i in range(data.f_obs.shaep[1])]

        self.data = data
        self.τ = data.τ
        self.t = data.t
        self.common_ind = data.common_indices.numpy()

    def plot_kinetics(self, k, k_f, true_k=None, true_k_f=None):
        k_latest = np.mean(k[-self.opt.num_kinetic_avg:], axis=0)
        num_genes = k.shape[1]
        num_tfs = k_f.shape[1]
        true_data = None

        hpds = list()
        for j in range(num_genes):
            hpds.append(arviz.hpd(k[-self.opt.num_hpd:, j,:], credible_interval=0.95))
        hpds = np.array(hpds)
        hpds = abs(hpds - np.expand_dims(k_latest, 2))

        width = 18 if num_genes > 10 else 10
        plt.figure(figsize=(width, 14))
        plt.suptitle('Transcription ODE Kinetic Parameters')
        comparison_label = 'True'
        if self.opt.plot_barenco:
            comparison_label = 'Barenco et al.'
            # From Martino paper ... do a rough rescaling so that the scales match.
            B_barenco = np.array([2.6, 1.5, 0.5, 0.2, 1.35])[[0, 4, 2, 3, 1]]
            B_barenco = B_barenco/np.mean(B_barenco)*np.mean(k_latest[:, 1])
            S_barenco = (np.array([3, 0.8, 0.7, 1.8, 0.7])/1.8)[[0, 4, 2, 3, 1]]
            S_barenco = S_barenco/np.mean(S_barenco)*np.mean(k_latest[:, 3])
            D_barenco = (np.array([1.2, 1.6, 1.75, 3.2, 2.3])*0.8/3.2)[[0, 4, 2, 3, 1]]
            D_barenco = D_barenco/np.mean(D_barenco)*np.mean(k_latest[:, 2])
            true_data = np.array([np.zeros(num_genes), B_barenco, D_barenco, S_barenco]).T
        elif true_k is not None:
            true_data = true_k
        plot_labels = ['Initial Conditions', 'Basal rates', 'Decay rates', 'Sensitivities']

        plotnum = 421
        for k in range(4):
            plt.subplot(plotnum)
            plotnum+=1
            plt.bar(np.arange(num_genes)-0.2, k_latest[:, k], width=0.4, tick_label=self.opt.gene_names, label='Model')
            if true_data is not None:
                plt.bar(np.arange(num_genes)+0.2, true_data[:, k], width=0.4, color='blue', align='center', label=comparison_label)
            plt.title(plot_labels[k])
            plt.errorbar(np.arange(num_genes)-0.2, k_latest[:, k], hpds[:, k].swapaxes(0,1), fmt='none', capsize=5, color='black')
            plt.legend()
            plt.xticks(rotation=80)
        k_latest = np.mean(k_f[-self.opt.num_kinetic_avg:], axis=0)

        plt.figure(figsize=(10, 6))
        plotnum = 221
        hpds = list()
        for i in range(num_tfs):
            hpds.append(arviz.hpd(k_f[-self.opt.num_hpd:, i,:], credible_interval=0.95))
        hpds = np.array(hpds)
        hpds = abs(hpds - np.expand_dims(k_latest, 2))
        for k in range(k_f.shape[2]):
            plt.subplot(plotnum)
            plotnum+=1
            plt.bar(np.arange(num_tfs)-0.1, k_latest[:, k], width=0.2, tick_label=self.opt.tf_names, label='Model')
            if true_k_f is not None:
                plt.bar(np.arange(num_tfs)+0.1, true_k_f[:, k], width=0.2, color='blue', align='center', label='True')
            plt.errorbar(np.arange(num_tfs)-0.1, k_latest[:, k], hpds[:, k].swapaxes(0,1), fmt='none', capsize=5, color='black')
            plt.legend()


    def plot_kinetics_convergence(self, k, k_f):
        num_genes = k.shape[1]
        labels = ['a', 'b', 'd', 's']
        height = (num_genes//3)*5
        plt.figure(figsize=(14, height))
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

        num_tfs = k_f.shape[1]
        width = 14 if num_tfs > 1 else 6
        plt.figure(figsize=(width, 4*np.ceil(num_tfs/2)))
        plt.suptitle('Translation Convergence')
        labels = ['a', 'δ']
        horizontal_subplots = 21 if num_tfs > 1 else 11
        for i in range(num_tfs):
            ax = plt.subplot(num_tfs*100 + horizontal_subplots + i)
            ax.set_title(f'TF {i}')
            for k in range(k_f.shape[2]):
                plt.plot(k_f[:, i, k], label=labels[k])
            plt.legend()


    def plot_genes(self, m_preds, replicate=0):
        m_preds = m_preds[:, replicate]
        plt.figure(figsize=(14, 17))
        plt.suptitle('Genes')
        num_genes = m_preds[0].shape[0]
        self.plot_samples(m_preds, self.data.m_obs[replicate], self.opt.gene_names, 
                          self.opt.num_plot_genes)

    def plot_samples(self, samples, scatters, titles, num_samples, color='grey', scatter_args={}):
        num_components = samples[0].shape[0]

        subplot_shape = (1, 1) if num_components < 2 else ((num_components+2)//3, 3)

        for j in range(num_components):
            ax = plt.subplot(subplot_shape[0], subplot_shape[1], 1+j)
            plt.title(titles[j])
            plt.scatter(self.τ[self.common_ind], scatters[j], marker='x', label='Observed', **scatter_args)
            # plt.errorbar([n*10+n for n in range(7)], Y[j], 2*np.sqrt(Y_var[j]), fmt='none', capsize=5)

            for s in range(1, num_samples):
                kwargs = {}
                if s == 1:
                    kwargs = {'label':'Samples'}

                plt.plot(self.τ, samples[-s,j,:], color=color, alpha=0.5, **kwargs)

            # HPD:
            bounds = arviz.hpd(samples[-self.opt.num_hpd:,j,:], credible_interval=0.95)
            plt.fill_between(self.τ, bounds[:, 0], bounds[:, 1], color='grey', alpha=0.3, label='95% credibility interval')

            plt.xticks(self.t)
            ax.set_xticklabels(self.t)

            plt.xlabel('Time (h)')
            plt.legend()
        plt.tight_layout()

    def plot_tfs(self, f_samples, replicate=0, scale_observed=False):
        f_samples = f_samples[:, replicate]
        num_tfs = self.data.f_obs.shape[1]
        plt.figure(figsize=(13, 7*np.ceil(num_tfs/2)))
        plt.suptitle('Transcription Factors')
        scatter_args = {'s': 60, 'linewidth': 2, 'color': 'tab:blue'}
        self.plot_samples(f_samples, self.data.f_obs[replicate], self.opt.tf_names, self.opt.num_plot_tfs,
                          color='cadetblue', scatter_args=scatter_args)
        # if 'σ2_f' in model.params._fields:
        #     σ2_f = model.params.σ2_f.value
        #     plt.errorbar(τ[common_indices], f_observed[0], 2*np.sqrt(σ2_f[0]), 
        #                 fmt='none', capsize=5, color='blue')
        # else:
        #     σ2_f = σ2_f_pre
        # for i in range(num_tfs):
        #     f_obs = self.data.f_obs[replicate, i]
        #     if scale_observed:
        #         f_obs = f_obs / np.mean(f_obs) * np.mean(f_i)

        if self.opt.plot_barenco:
            barenco_f, _ = scaled_barenco_data(np.mean(f_samples[-10:], axis=0))
            plt.scatter(self.τ[self.common_ind], barenco_f, marker='x', s=60, linewidth=3, label='Barenco et al.')

        plt.tight_layout()

    def plot_noises(self, σ2_m_samples, σ2_f_samples):
        plt.figure(figsize=(5, 3))
        for j in range(self.opt.gene_names.shape[0]):
            plt.plot(σ2_m_samples[:, j], label=self.opt.gene_names[j])
        plt.legend()
        if σ2_f_samples is not None:
            plt.figure(figsize=(5, 3))
            for j in range(σ2_f_samples.shape[1]):
                plt.plot(σ2_f_samples[:, j])
            plt.legend()

    def plot_weights(self, weights):
        plt.figure()
        w = weights[0]
        w_0 = weights[1]
        for j in range(self.opt.gene_names.shape[0]):
            plt.plot(w[:, j], label=self.opt.gene_names[j])
        plt.legend()
        plt.title('Interaction weights')
        plt.figure()
        for j in range(self.opt.gene_names.shape[0]):
            plt.plot(w_0[:,j])
        plt.title('Interaction bias')


    def generate_report(self, results: SampleResults, m_preds,
                        true_k=None,
                        true_k_f=None,
                        replicate=0,
                        scale_observed=False):


        self.plot_tfs(results.f, replicate=replicate, scale_observed=scale_observed)

        self.plot_genes(m_preds, replicate=replicate)

        self.plot_kinetics_convergence(results.k, results.k_f)

        self.plot_kinetics(results.k, results.k_f, true_k=true_k, true_k_f=true_k_f)
                    
        plt.figure(figsize=(10, 4))
        plotnum = 0
        for name, param in (zip(['Param 1', 'Param 2'], results.kernel_params)):
            ax = plt.subplot(221+plotnum)
            plt.plot(param)
            ax.set_title(name)
            plotnum+=1

        self.plot_noises(results.σ2_m, results.σ2_f)

        if results.weights is not None:
            self.plot_weights(results.weights)