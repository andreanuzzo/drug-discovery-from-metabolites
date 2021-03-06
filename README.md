[![DOI](https://zenodo.org/badge/269442521.svg)](https://zenodo.org/badge/latestdoi/269442521)

# Expanding the drug discovery space
This is the code repository for the paper `Expanding the drug discovery space with predicted metabolite-target interactions` by Nuzzo, A. et al. 

The aim of this project is to determine novel targets for Inflammatory Bowel Disease (IBD), i.e. Crohn's disease (CD) and/or Ulcerative Colitis (UC) from publicly available microbiome raw_data. The iHMP project (NIDDK U54DE023798) is conducted by several institutions, including Broad Institute and Stanford University. 

The repository contains:
1. Steps to reproduce the environment 
2. Look-up tables used for the analysis
3. Steps to reproduce the analysis and the figures 


## Data
The elaborated data and sample data from the HMP2 original [IBD study](https://doi.org/10.1038/s41586-019-1237-9) are hosted by the [IBDMDB portal](https://ibdmdb.org).
- The LC/MS spectra for the metabolomics analyses are also available, but here we will work on the pre-processed metabolite abundances computed using the [MetaboAnalyst 2](https://doi.org/10.3390/metabo9030057) software against the [HMDB](https://doi.org/10.1093/nar/gkx1089) raw_database. Original HMP2 protocol is available [here](https://www.ibdmdb.org/cb/document/Data%20Generation%20Protocols/MetabolomicsHMP2Protocol.pdf).
- The host transcriptomics data are obtained via bulk RNASeq from biopsies of patients in different location of the intestines. The Transcripts were aligned to hg19 using topHat 2.0.14 and resolved to counts with htseq-count 0.6.1. Here again we'll work with the pre-processed tables.Original HMP2 protocol is available [here](https://www.ibdmdb.org/cb/document/Data%20Generation%20Protocols/Host_Transcriptomics_HMP2_protocol.pdf).

## Databases:
- Metabolites annotation is parsed from the [HMDB database](https://hmdb.ca) (as of May 2020)
- Functional assay data are parsed from [ChEMBL v.25](https://www.ebi.ac.uk/chembl/). Those are not parsed dynamically, but elaborated files are stored
- Genetic association data are parsed from [GWAS catalog](https://www.ebi.ac.uk/gwas/)(as of July 14, 2020) and [OMIM](https://www.omim.org/) (as of May 28, 2020)

## Steps to reproduce the analysis:

### 1. Clone the repository 

Remember to change the argument `$PATH_TO_YOUR_DIRECTORY`as desired (required also for the following steps)

```
git clone https://github.com/andreanuzzo/drug-discovery-from-metabolites.git $PATH_TO_YOUR_DIRECTORY
```
### 2. Run the code

#### 2.1. Docker 
If you have docker, you can easily run the script as follows, without doing any installation. The docker is based on [rocker/rstudio:3.6](https://www.rocker-project.org) and contains all the packages for installation. You can run the code as follows.
```
cd $PATH_TO_YOUR_DIRECTORY

docker run \
-v "$(pwd):/home/rstudio/MMIM" \
-e DISABLE_AUTH=true \
-e ROOT=TRUE \
-p 8787:8787 \
--rm andreanuzzo/mmim
```


#### 2.2. Local computer 
If you don't want to use docker, follow the next steps to run the code locally. You will need to have Python, RStudio and virtual environments pre-installed.

Prepare the environment, changing the argument `$PATH_TO_YOUR_DIRECTORY`
```
## Note: requires  R <= 3.6.2 and Python => 3.6.0
cd $PATH_TO_YOUR_DIRECTORY

## R packages
RENV_VERSION=0.12.0
R -e "install.packages('remotes', repos = c(CRAN = 'https://cloud.r-project.org'))"
R -e "remotes::install_github('rstudio/renv@${RENV_VERSION}')"
R -e 'renv::restore()'

## Python environment (requires venv)
python3 -m venv .venv
source .venv/bin/activate
pip3 install numpy==1.18.1 matplotlib==3.1.1 pandas==0.25.3 wheel
pip3 install -r requirements.txt
deactivate
```

Get data, changing the argument `$PATH_TO_YOUR_DIRECTORY`
```
bash scripts/get_data.sh $PATH_TO_YOUR_DIRECTORY
```

Run the notebooks as follows, changing the argument `$PATH_TO_YOUR_DIRECTORY`
```
bash scripts/run_analysis.sh $PATH_TO_YOUR_DIRECTORY
```
### 3. Output
Regardless of the way you choose to run the code, figures and tables from the manuscript should be reproduced in `$PATH_TO_YOUR_DIRECTORY/results/manuscript_files` as follows:

| File                       	  | Description                                                                                                |
|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------	|
| **Fig.1.pdf**              	  | Metabolomics analysis                                                                                                          	|
| **Fig.2.pdf**             	  | Metabolite - Target connections based upon metabolomics analysis                                                               	|
| **Fig.3.pdf**              	  | Host transcriptomics analysis results and Target - Metabolite connections based upon them                                      	|
| **Fig.4.pdf**              	  | GGraph version of Target - Metabolite connection for NOS2, HTR4, GABRG2, SLC22A3 (note: manuscript used the Cytoscape version) 	|
| **Fig.5a.pdf**             	  | BioMAP results for the selected metabolites (Fig 5b was done in [MetaCore™](https://portal.genego.com/cgi/data_manager.cgi#))  	|
| **Supplementary Fig.1.pdf**	  | Overview of Metabolomics results for each analysis method                                                                      	|
| **Supplementary Fig.2.pdf**	  | Overview of consensus scoring across metabolomics analysis methods                                                             	|
| **Supplementary Fig.3.pdf**	  | Metabolite scoring classification per each metabolite source (as per HMDB classification)                                      	|
| **Supplementary Fig.4.pdf**	  | Host transcriptomics analysis results                                                                                          	|
| **Supplementary Fig.6.pdf**	  | Full BioMAP results for the selected metabolites                                                                               	|
| **Supplementary_Table1.xlsx** | Full table of metabolomics analysis results                                                                                    	|
| **Supplementary_Table2.xlsx** | Full table of Metabolite - Target connections based upon metabolomics analysis                                                 	|
| **Supplementary_Table3.xlsx** | Full table of Target - Metabolite connections based upon host transcriptomics analysis                                         	|
| **Supplementary_Table4.xlsx** | Full table of Target - Metabolite connections based upon genetic association to diseases                                       	|
| **Supplementary_Table5.xlsx** | Full BioMAP results for the selected metabolites                                                                               	|
| **Supplementary_Table6.xlsx** | Full table of Metabolite-Target associations with high confidence                                                              	|
