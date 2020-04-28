import csv
import pandas as pd
import numpy as np
from sklearn import preprocessing

def load_3day_dros():
    with open('data/3day/GSE47999_Normalized_Counts.txt', 'r', 1) as f:
        contents = f.buffer
        df = pd.read_table(contents, sep='\t', index_col=0)
    replicates = 3
    columns = df.columns[df.columns.str.startswith('20,000')][::replicates]
    known_target_genes = ['FBgn0011774', 'FBgn0030189', 'FBgn0031713', 'FBgn0032393', 'FBgn0037020', 'FBgn0051864']
    tf_names = ['FBgn0039044']
    genes_df = df[df.index.isin(known_target_genes)][columns]
    tfs_df = df[df.index.isin(tf_names)][columns]

    #Normalise across time points
    normalised = preprocessing.normalize(np.r_[genes_df.values,tfs_df.values])
    genes = normalised[:-1]
    tfs = np.atleast_2d(normalised[-1])
    return (genes_df, np.float64(genes)), (tfs_df, np.float64(tfs)), np.array([2, 10, 20])


def load_barenco_puma():
    mmgmos_processed = True
    if mmgmos_processed:
        with open('data/barencoPUMA_exprs.csv', 'r') as f:
            df = pd.read_csv(f, index_col=0)
        with open('data/barencoPUMA_se.csv', 'r') as f:
            dfe = pd.read_csv(f, index_col=0)
        columns = [f'cARP1-{t}hrs.CEL' for t in np.arange(7)*2]
    else:
        with open('data/barenco_processed.tsv', 'r') as f:
            df = pd.read_csv(f, delimiter='\t', index_col=0)

        columns = [f'H_ARP1-{t}h.3' for t in np.arange(7)*2]

    known_target_genes = ['203409_at', '202284_s_at', '218346_s_at', '205780_at', '209295_at', '211300_s_at']
    genes = df[df.index.isin(known_target_genes)][columns]
    genes_se = dfe[dfe.index.isin(known_target_genes)][columns]

    assert df[df.duplicated()].size == 0

    index = {
        '203409_at': 'DDB2',
        '202284_s_at': 'p21',
        '218346_s_at': 'SESN1',
        '205780_at': 'BIK',
        '209295_at': 'TNFRSF10b',
        '211300_s_at': 'p53'
    }
    genes.rename(index=index, inplace=True)
    genes_se.rename(index=index, inplace=True)

    # Reorder genes
    df_genes = genes.reindex(['DDB2', 'SESN1', 'TNFRSF10b', 'p21', 'BIK', 'p53'])
    genes = df_genes.values
    genes_se = genes_se.reindex(['DDB2', 'SESN1', 'TNFRSF10b', 'p21', 'BIK', 'p53']).values

    Y_var = genes_se*genes_se
    Y = np.exp(genes+Y_var/2)
    Y_var = (np.exp(Y_var)-1) * np.exp(2*genes + Y_var)

    scale = np.sqrt(np.var(Y, axis=1))
    scale_mat = np.c_[[scale for _ in range(7)]].T
    Y = Y / scale_mat
    Y_var = Y_var / (scale_mat * scale_mat)

    m_observed = np.float64(Y[:-1])
    f_observed = np.float64(np.atleast_2d(Y[-1]))
    σ2 = np.float64(Y_var[:-1])
    σ2_f = np.float64(np.atleast_2d(Y_var[-1]))

    return df_genes, genes, genes_se, m_observed, f_observed, Y_var, σ2, σ2_f, np.arange(7)*2           # Observation times
