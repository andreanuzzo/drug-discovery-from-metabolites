import os
import pandas as pd
import xml.etree.ElementTree as ET

global wdir
wdir = os.getcwd()

e = ET.parse(os.path.join(wdir, 'raw_data/hmdb_metabolites.xml')).getroot()

nodelist = [element.tag for element in e.iter()]

HMDB = []
Secondary = []
Metabolites = []
direct_parent = []
sub_class = []
main_class = []
super_class = [] 
SMILES = []
KEGG = []
gene_name = []
Sources=[]

for i, code in  enumerate(e.iter('{http://www.hmdb.ca}metabolite')):
#  if i <10:
    try:
      gene_name.append([el.text for el in \
                code.findall('{http://www.hmdb.ca}protein_associations/{http://www.hmdb.ca}protein/{http://www.hmdb.ca}gene_name')])
    except IndexError:
      gene_name.append(None)
for i, code in  enumerate(e.iter('{http://www.hmdb.ca}metabolite')):
#  if i <10:
    try:
      KEGG.append([el.text for el in \
                code.findall('{http://www.hmdb.ca}kegg_id')][0])
    except IndexError:
      KEGG.append('Not Available')
    HMDB.append(code.find('{http://www.hmdb.ca}accession').text)
    try:
      Secondary.append([el.text for el in \
             code.findall('{http://www.hmdb.ca}secondary_accessions/{http://www.hmdb.ca}accession')])
    except AttributeError:
      Secondary.append(code.find('{http://www.hmdb.ca}accession').text)
    try:
      Metabolites.append([el.text for el in \
                code.findall('{http://www.hmdb.ca}name')][0])
    except IndexError:
      Metabolites.append('Not Available')
    try:
      direct_parent.append(code.find('{http://www.hmdb.ca}taxonomy').\
                            find('{http://www.hmdb.ca}direct_parent').text)
    except AttributeError:
      direct_parent.append('Not Available')
    try:
      sub_class.append(code.find('{http://www.hmdb.ca}taxonomy').\
                            find('{http://www.hmdb.ca}sub_class').text)
    except AttributeError:
      sub_class.append('Not Available')
    try:
      main_class.append(code.find('{http://www.hmdb.ca}taxonomy').\
                            find('{http://www.hmdb.ca}class').text)
    except AttributeError:
      main_class.append('Not Available')
    try:
      super_class.append(code.find('{http://www.hmdb.ca}taxonomy').\
                            find('{http://www.hmdb.ca}super_class').text)
    except AttributeError:
      super_class.append('Not Available')
    try:
      SMILES.append(code.find('{http://www.hmdb.ca}smiles').text)
    except AttributeError:
      SMILES.append('Not Available')
    try: 
      fall = code.findall('{http://www.hmdb.ca}ontology/{http://www.hmdb.ca}root/{http://www.hmdb.ca}descendants/{http://www.hmdb.ca}descendant')
      if len(fall)==0:
        Sources.append(['Not Available'])
      elif any(desc.find('{http://www.hmdb.ca}term').text=='Source' for desc in fall):
        for desc in fall:
          if desc.find('{http://www.hmdb.ca}term').text=='Source':
            fins = []
            for child in desc.find('{http://www.hmdb.ca}descendants'):
              if child.find('{http://www.hmdb.ca}term').text=='Endogenous':
                fins.append('Endogenous')
              elif child.find('{http://www.hmdb.ca}term').text=='Biological':
                for fin in child.find('{http://www.hmdb.ca}descendants'):
                  fins.append(fin.find('{http://www.hmdb.ca}term').text)
            Sources.append(fins)
      else:
        Sources.append(['Not available'])
    except IndexError:
      Sources.append('Not Available')


hmdb_ref = pd.DataFrame()
hmdb_ref['HMDB'] = HMDB
hmdb_ref['HMDB_short'] = Secondary
hmdb_ref['Metabolites'] = Metabolites
hmdb_ref['direct_parent'] = direct_parent
hmdb_ref['sub_class'] = sub_class
hmdb_ref['main_class'] = main_class
hmdb_ref['super_class'] = super_class
hmdb_ref['SMILES'] = SMILES
hmdb_ref['KEGG'] = KEGG
hmdb_ref['gene_name'] = gene_name
hmdb_ref['sources'] = Sources

temp = pd.DataFrame.from_records(hmdb_ref.HMDB_short.tolist())\
                .stack()\
                .reset_index(level=1, drop=True)\
                .rename('Alternative')
temp2 = pd.DataFrame.from_records(hmdb_ref.sources.tolist())\
                .stack()\
                .reset_index(level=1, drop=True)\
                .rename('origin')
temp3 = pd.DataFrame.from_records(hmdb_ref.gene_name.tolist())\
                .stack()\
                .reset_index(level=1, drop=True)\
                .rename('hgncid')

hmdb_ref = hmdb_ref.drop('HMDB_short', axis=1)\
            .join(temp)\
  .join(temp2)\
  .reset_index(drop=True)[['HMDB','Alternative','Metabolites',
                           'direct_parent','sub_class',
                           'main_class','super_class',
                           'origin','SMILES','KEGG'#,'hgncid'
                          ]]
    
#hmdb_ref.to_csv('lookup_tables/hmdb_ref_w_hgncid.csv', index=False)
hmdb_ref.drop_duplicates().to_csv(os.path.join(wdir,'tmp/hmdb_ref_w_origin.csv'), index=False)
