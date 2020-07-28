#!/bin/bash

mkdir raw_data
mkdir lookup_tables

#Sample data
wget https://ibdmdb.org/tunnel/products/HMP2/Metadata/hmp2_metadata.csv -O raw_data/hmp2_metadata.csv

#Metabolomics tables
wget https://ibdmdb.org/tunnel/products/HMP2/Metabolites/1723/HMP2_metabolomics.csv.gz -O raw_data/HMP2_metabolomics.csv.gz

#Host transcriptomes
wget https://ibdmdb.org/tunnel/products/HMP2/HTX/1730/host_tx_counts.tsv.gz -O raw_data/host_tx_counts.tsv.gz

#HMDB database
wget http://www.hmdb.ca/system/downloads/current/hmdb_metabolites.zip -O raw_data/HMDB.zip
unzip raw_data/HMDB.zip -d raw_data

source .venv/bin/activate && python3 scripts/HMDB_parser.py && deactivate

#Supplementary Tables from Lloyd-Price et al. 2020
wget -q -O raw_data/.tmp.zip https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-019-1237-9/MediaObjects/41586_2019_1237_MOESM6_ESM.zip 
unzip raw_data/.tmp.zip -d elaborated_data
