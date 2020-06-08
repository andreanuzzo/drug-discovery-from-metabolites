#!/bin/sh

mkdir raw_data

#Sample data
wget https://ibdmdb.org/tunnel/products/HMP2/Metadata/hmp2_metadata.csv -O raw_data/hmp2_metadata.csv

#Metabolomics tables
wget https://ibdmdb.org/tunnel/products/HMP2/Metabolites/1723/HMP2_metabolomics.csv.gz -O raw_data/HMP2_metabolomics.csv.gz

#Host transcriptomes
wget https://ibdmdb.org/tunnel/products/HMP2/HTX/1730/host_tx_counts.tsv.gz -O raw_data/host_tx_counts.tsv.gz

#HMDB database
wget http://www.hmdb.ca/system/downloads/current/hmdb_metabolites.zip -O raw_data/HMDB.zip
unzip raw_data/HMDB.zip

source .venv/bin/activate && python3 scripts/HMDB_parser.py && deactivate
