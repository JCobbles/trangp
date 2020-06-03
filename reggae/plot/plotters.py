from matplotlib import pyplot as plt
import numpy as np
import arviz
from reggae.data_loaders import scaled_barenco_data
from reggae.models.results import SampleResults
from dataclasses import dataclass, field
from matplotlib.animation import FuncAnimation

@dataclass
class PlotOptions:
    num_plot_genes: int = 20
    num_plot_tfs:   int = 20
    gene_names:     list = None
    tf_names:       list = None
    plot_barenco:   bool = False
    num_kinetic_avg:int = 50
    num_hpd:        int = 100
    true_label:     str = 'True'
    tf_present:     bool = True
    for_report:     bool = True
    ylabel:         str = ''
    protein_present:bool = True
    kernel_names:   list = field(default_factory=lambda:['Param 1', 'Param 2']) 

# From Martino paper ... do a rough rescaling so that the scales match.
barenco = np.stack([
    np.array([2.6, 1.5, 0.5, 0.2, 1.35])[[0, 4, 2, 3, 1]],
    (np.array([1.2, 1.6, 1.75, 3.2, 2.3])*0.8/3.2)[[0, 4, 2, 3, 1]],
    (np.array([3, 0.8, 0.7, 1.8, 0.7])/1.8)[[0, 4, 2, 3, 1]]
]).T

class Plotter():
    def __init__(self, data, options: PlotOptions):
        self.opt = options
        if self.opt.gene_names is None:
            self.opt.gene_names = np.arange(data.m_obs.shape[1])
        if self.opt.tf_names is None:
            self.opt.tf_names = [f'TF {i}' for i in range(data.f_obs.shape[1])]

        self.data = data
        self.τ = data.τ
        self.t = data.t
        self.common_ind = data.common_indices.numpy()

    def plot_kinetics(self, k, k_f, true_k=None, true_k_f=None):
        k_latest = np.mean(k[-self.opt.num_kinetic_avg:], axis=0)
        num_genes = k.shape[1]
        true_data = None
        plot_labels = ['Initial Conditions', 'Basal rates', 'Decay rates', 'Sensitivities']

        hpds = list()
        for j in range(num_genes):
            hpds.append(arviz.hpd(k[-self.opt.num_hpd:, j,:], credible_interval=0.95))
        hpds = np.array(hpds)
        hpds = abs(hpds - np.expand_dims(k_latest, 2))

        width = 18 if num_genes > 10 else 10
        plt.figure(figsize=(width, 16))
        if not self.opt.for_report:
            plt.suptitle('Transcription ODE Kinetic Parameters')
        comparison_label = self.opt.true_label
        if self.opt.plot_barenco:
            comparison_label = 'Barenco et al.'
            if self.opt.protein_present:
                true_data = barenco / np.mean(barenco, axis=0) * np.mean(k_latest[1:], axis=0)
                true_data = np.c_[np.zeros(num_genes), true_data]
            else: 
                true_data = barenco / np.mean(barenco, axis=0) * np.mean(k_latest, axis=0)
                plot_labels = plot_labels[1:]
        elif true_k is not None:
            true_data = true_k

        plotnum = 421
        for k in range(k_latest.shape[1]):
            plt.subplot(plotnum)
            plotnum+=1
            plt.bar(np.arange(num_genes)-0.2, k_latest[:, k], width=0.4, tick_label=self.opt.gene_names, label='Model')
            if true_data is not None:
                plt.bar(np.arange(num_genes)+0.2, true_data[:, k], width=0.4, color='blue', align='center', label=comparison_label)
            plt.title(plot_labels[k])
            plt.errorbar(np.arange(num_genes)-0.2, k_latest[:, k], hpds[:, k].swapaxes(0,1), fmt='none', capsize=5, color='black')
            plt.legend()
            plt.xticks(rotation=70)
        plt.tight_layout(h_pad=2.0)
        if self.opt.protein_present:
            k_latest = np.mean(k_f[-self.opt.num_kinetic_avg:], axis=0)
            plt.figure(figsize=(10, 6))
            self.plot_bar_hpd(k_f, k_latest, self.opt.tf_names, true_var=true_k_f)
    
    def plot_bar_hpd(self, var_samples, var, labels, true_var=None): # var = (n, m) num == n == #bars, m #plots
        hpds = list()
        num = var.shape[0]
        plotnum = var.shape[1]*100 + 21
        for i in range(num):
            hpds.append(arviz.hpd(var_samples[-self.opt.num_hpd:, i,:], credible_interval=0.95))
        hpds = np.array(hpds)
        hpds = abs(hpds - np.expand_dims(var, 2))
        for k in range(var_samples.shape[2]):
            plt.subplot(plotnum)
            plotnum+=1
            plt.bar(np.arange(num)-0.1, var[:, k], width=0.2, tick_label=labels, label='Model')
            plt.errorbar(np.arange(num)-0.1, var[:, k], hpds[:, k].swapaxes(0,1), fmt='none', capsize=5, color='black')
            plt.xlim(-1, num)
            if true_var is not None:
                plt.bar(np.arange(num)+0.1, true_var[:, k], width=0.2, color='blue', align='center', label='True')
                plt.legend()
        plt.tight_layout()

    def plot_kinetics_convergence(self, k, k_f):
        num_genes = k.shape[1]
        labels = ['a', 'b', 'd', 's']
        height = (num_genes//2)*5
        plt.figure(figsize=(14, height))
        plt.suptitle('Convergence of ODE Kinetic Parameters')
        self.plot_kinetics_convergence_group(k, labels, 'Gene')
        if self.opt.protein_present:
            width = 14 if k_f.shape[1] > 1 else 6
            plt.figure(figsize=(width, 4*np.ceil(k_f.shape[1]/2)))
            plt.suptitle('Translation Convergence')
            labels = ['a', 'δ']
            self.plot_kinetics_convergence_group(k_f, labels, 'TF')

    def plot_kinetics_convergence_group(self, k, labels, title_prefix):
        num = k.shape[1]
        horizontal_subplots = min(2, num)
        for j in range(num):
            ax = plt.subplot(num, horizontal_subplots, j+1)
            ax.set_title(f'{title_prefix} {j}')            
            for v in range(k.shape[2]):
                plt.plot(k[:, j, v], label=labels[v])
            plt.legend()
        plt.tight_layout()

    def plot_samples(self, samples, titles, num_samples, color='grey', scatters=None, 
                     scatter_args={}, legend=True, margined=False):
        num_components = samples[0].shape[0]
        subplot_shape = ((num_components+2)//3, 3)
        if self.opt.for_report:
            subplot_shape = ((num_components+1)//2, 2)
        if num_components <= 1:
            subplot_shape = (1, 1) 
        for j in range(num_components):
            ax = plt.subplot(subplot_shape[0], subplot_shape[1], 1+j)
            plt.title(titles[j])
            if scatters is not None:
                plt.scatter(self.τ[self.common_ind], scatters[j], marker='x', label='Observed', **scatter_args)
            # plt.errorbar([n*10+n for n in range(7)], Y[j], 2*np.sqrt(Y_var[j]), fmt='none', capsize=5)

            for s in range(1, num_samples):
                kwargs = {}
                if s == 1:
                    kwargs = {'label':'Samples'}

                plt.plot(self.τ, samples[-s,j,:], color=color, alpha=0.5, **kwargs)
            if j % subplot_shape[1] == 0:
                plt.ylabel(self.opt.ylabel)
            # HPD:
            bounds = arviz.hpd(samples[-self.opt.num_hpd:,j,:], credible_interval=0.95)
            plt.fill_between(self.τ, bounds[:, 0], bounds[:, 1], color='grey', alpha=0.3, label='95% credibility interval')

            plt.xticks(self.t)
            ax.set_xticklabels(self.t)
            if margined:
                plt.ylim(min(samples[-1, j])-2, max(samples[-1, j]) + 2)
            plt.xlabel('Time (h)')
            if legend:
                plt.legend()
        plt.tight_layout()

    def plot_genes(self, m_preds, replicate=0, height_mul=3, width_mul=2):
        m_preds = m_preds[:, replicate]
        height = np.ceil(m_preds.shape[1]/3)
        width = 4 if self.opt.for_report else 7
        plt.figure(figsize=(width*width_mul, height*height_mul))
        if not self.opt.for_report:
            plt.suptitle('Genes')
        self.plot_samples(m_preds, self.opt.gene_names, self.opt.num_plot_genes, 
                          scatters=self.data.m_obs[replicate], legend=not self.opt.for_report)

    def plot_tfs(self, f_samples, replicate=0, scale_observed=False):
        f_samples = f_samples[:, replicate]
        num_tfs = self.data.f_obs.shape[1]
        width = 8 if num_tfs > 1 else 6
        plt.figure(figsize=(width, 4*np.ceil(num_tfs/3)))
        scatter_args = {'s': 60, 'linewidth': 2, 'color': 'tab:blue'}
        self.plot_samples(f_samples, self.opt.tf_names, self.opt.num_plot_tfs, 
                          scatters=self.data.f_obs[replicate] if self.opt.tf_present else None,
                          color='cadetblue', scatter_args=scatter_args, margined=True)
        
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

    def anim_latent(self, results, index=0, replicate=0):
        fig = plt.figure()
        ax = plt.axes(xlim=(-1, 13), ylim=(-1, 5))
        line, = ax.plot([], [], lw=3)
        f_samples = results.f
        def init():
            line.set_data([], [])
            return line,
        def animate(i):
            x = np.linspace(0, 12, f_samples.shape[3])
            y = f_samples[i*10, replicate, index]
            line.set_data(x, y)
            return line,
        anim = FuncAnimation(fig, animate, init_func=init,
                                    frames=f_samples.shape[0]//10, interval=50, blit=True)
        return anim.to_html5_video()

    def summary(self, results: SampleResults, m_preds, true_k=None, true_k_f=None,
                replicate=0, scale_observed=False):
        self.plot_tfs(results.f, replicate=replicate, scale_observed=scale_observed)
        self.plot_genes(m_preds, replicate=replicate)
        self.plot_kinetics(results.k, results.k_f, true_k=true_k, true_k_f=true_k_f)
        plt.figure()
        kp = np.array(results.kernel_params).swapaxes(0,1)
        kp_latest = np.mean(kp[-50:], axis=0)
        self.plot_bar_hpd(kp, kp_latest, self.opt.kernel_names)

    def convergence_summary(self, results: SampleResults):
        self.plot_kinetics_convergence(results.k, results.k_f)
        plt.figure(figsize=(10, 4))
        plotnum = 0
        for name, param in zip(self.opt.kernel_names, results.kernel_params):
            ax = plt.subplot(221+plotnum)
            plt.plot(param)
            ax.set_title(name)
            plotnum+=1

        self.plot_noises(results.σ2_m, results.σ2_f)
        if results.weights is not None:
            self.plot_weights(results.weights)
