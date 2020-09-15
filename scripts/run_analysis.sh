#!/bin/sh
basepath=$(echo "$1/$line")

cd $basepath

#Rscript -e "renv::restore()"

echo "Analyzing metabolomics data\n"
Rscript -e "rmarkdown::render('scripts/Metabolomics.Rmd')"

echo "Analyzing host transcriptomics data\n"
Rscript -e "rmarkdown::render('scripts/Host_transcriptomics.Rmd')"

echo "Reproducing figures and tables\n"
Rscript -e "rmarkdown::render('scripts/Multiomics_and_figures.Rmd')"

echo "Done"
