# Expanding the drug discovery space
This is the code repository for the paper `blabla`

The aim of this project is to determine novel targets for Inflammatory Bowel Disease (IBD), i.e. Crohn's disease (CD) and/or Ulcerative Colitis (UC) from publicly available microbiome raw_data. The iHMP project (NIDDK U54DE023798) is conducted by several institutions, including Broad Institute and Stanford University. 

# Data
The elaborated data and sample data from the HMP2 original [IBD study](https://doi.org/10.1038/s41586-019-1237-9) are hosted by the [IBDMDB portal](https://ibdmdb.org).
- The LC/MS spectra for the metabolomics analyses are also available, but here we will work on the pre-processed metabolite abundances computed using the [MetaboAnalyst 2](https://doi.org/10.3390/metabo9030057) software against the [HMDB](https://doi.org/10.1093/nar/gkx1089) raw_database.
- The host transcriptomics data are obtained via bulk RNASeq from biopsies of patients in different location of the intestines. The Transcripts were aligned to hg19 using topHat 2.0.14 and resolved to counts with htseq-count 0.6.1. Here again we'll work with the pre-processed tables.

# Databases:
- Metabolites annotation is parsed from the [HMDB database](https://hmdb.ca) (as of May 2020)
- Functional assay data are parsed from [ChEMBL v.25](https://www.ebi.ac.uk/chembl/). Those are not parsed dynamically, but elaborated files are stored
- Genetic association data are parsed from [GWAS catalog](https://www.ebi.ac.uk/gwas/)(dynamically) and [OMIM](https://www.omim.org/) (pre-processed)

The repository contains:
1. Steps to reproduce the environment `#TODO:docker`
2. Links to the original data and the lookup tables used for the analysis
3. Steps to reproduce the analysis and the figures 

