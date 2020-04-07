import csv
import pandas as pd
import numpy as np

def load_barenco_puma():
    mmgmos_processed = True
    if mmgmos_processed:
        with open('barencoPUMA_exprs.csv', 'r') as f:
            df = pd.read_csv(f, index_col=0)
        with open('barencoPUMA_se.csv', 'r') as f:
            dfe = pd.read_csv(f, index_col=0)
        columns = [f'cARP1-{t}hrs.CEL' for t in np.arange(7)*2]
    else:
        with open('barenco_processed.tsv', 'r') as f:
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
    genes = genes.reindex(['DDB2', 'SESN1', 'TNFRSF10b', 'p21', 'BIK', 'p53']).values
    genes_se = genes_se.reindex(['DDB2', 'SESN1', 'TNFRSF10b', 'p21', 'BIK', 'p53']).values

    Y_var = genes_se*genes_se
    Y = np.exp(genes+Y_var/2)
    Y_var = (np.exp(Y_var)-1) * np.exp(2*genes + Y_var)

    scale = np.sqrt(np.var(Y, axis=1));
    scale_mat = np.c_[[scale for _ in range(7)]].T
    Y = Y / scale_mat
    Y_var = Y_var / (scale_mat * scale_mat)

    return df, genes, genes_se, Y, Y_var
