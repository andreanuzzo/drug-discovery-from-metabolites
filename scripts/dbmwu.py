import numpy as np
import pandas as pd
import dabest

def dbmwu(data, category='diagnosis', indexes=['nonIBD','UC','CD']):
  results = pd.DataFrame()
  for feature in data.columns[1:-1]:
    test = dabest.load(data=data.fillna(0),
                x=category,
                y=feature,
                idx=indexes,
                random_seed=42)

    feat_res = test.\
    hedges_g.statistical_tests.\
    pivot_table(index='control', 
          columns='test',
          values=['difference', 'pvalue_mann_whitney']).\
    reset_index()

    feat_res.columns = ['_'.join(col).strip() for col in feat_res.columns.values]
    feat_res.iloc[:,0]=feature
    feat_res.rename(columns={'control_':'HMDB'}, inplace=True)

    results = pd.concat([results, feat_res])
    
  return results
