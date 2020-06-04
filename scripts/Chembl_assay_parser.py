import numpy as np
import os
import pandas as pd
import pyodbc

global wdir
cwdir = os.getcwd()
wdir = os.path.join(cwdir, os.pardir)

#NOTE: similarity searches were conducted in the SmiScreen page with
#Tanimoto >.8 and Tversky >.9, a=0.5 against GSKSTores and Chembl25

Chembl25_tanimoto = pd.read_csv(os.path.join(wdir,'elaborated_data/CHEMBL25.CFP.csv.gz'))
Chembl25_tversky05 = pd.read_csv(os.path.join(wdir,'elaborated_data/CHEMBL25.CFP_Tv05.csv.gz'))

conn = pyodbc.connect("DSN=impaladsn", autocommit=True)

Chembl25_tanimoto['Similarity_type']='Tanimoto'
Chembl25_tversky05['Similarity_type']='Tversky_sub'

chembl_query = Chembl25_tanimoto.\
  rename(columns={'tarName':'HMDB'}).\
  loc[:,['HMDB','Similarity','Similarity_type','ChemblID']].\
  merge(Chembl25_tversky05.\
  rename(columns={'tarName':'HMDB'}).\
  loc[:,['HMDB','Similarity','Similarity_type','ChemblID']],
       how='left', on=['HMDB','ChemblID'])
  
chembl_ids = np.array_split(chembl_query.ChemblID.unique(),200)
chembl_assays = pd.DataFrame()

for arr in chembl_ids:
  arrq = tuple(arr)
  query = f"""SELECT DISTINCT m.chembl_id AS compound_chembl_id,
                a.description AS assay_description,
                act.standard_type,
                act.bao_endpoint as bao,
                act.pchembl_value,
                t.chembl_id AS target_chembl_id,
                cs.component_synonym AS hgncid,
                t.pref_name AS target_name
  FROM assay_chembl_use.molecule_dictionary AS m
  LEFT OUTER JOIN assay_chembl_use.compound_records AS r ON m.molregno = r.molregno
  LEFT OUTER JOIN assay_chembl_use.activities AS act ON r.record_id = act.record_id
  LEFT OUTER JOIN assay_chembl_use.assays AS a ON act.assay_id = a.assay_id
  LEFT OUTER JOIN assay_chembl_use.target_dictionary AS t ON a.tid = t.tid
  LEFT OUTER JOIN assay_chembl_use.target_components AS tc ON t.tid = tc.tid
  LEFT OUTER JOIN assay_chembl_use.component_synonyms AS cs ON tc.component_id = cs.component_id
  WHERE m.chembl_id IN {arrq}
  AND t.organism = 'Homo sapiens'
  AND cs.syn_type = 'GENE_SYMBOL'
  AND act.type IS NOT NULL
  AND a.assay_type = 'F'"""

  chembl_assays= pd.concat([chembl_assays, pd.read_sql(query, conn)],
                        axis = 0)
  
chembl_assays.drop_duplicates().to_csv(os.path.join(wdir,'elaborated_data/Chembl_f_assays_BIOME.csv', index=False))
chembl_assays.head().T
chembl_assays.shape

