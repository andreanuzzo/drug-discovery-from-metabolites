#!/bin/sh
basepath=$(echo "$1/$line")

cd $basepath

echo "Analyzing metabolomics data\n"
Rscript -e "rmarkdown::render('scripts/Metabolomics.Rmd',params=list(basepath = $basepath))"

echo "Analyzing host transcriptomics data\n"
Rscript -e "rmarkdown::render('scripts/Host_transcriptomics.Rmd',params=list(basepath = basepath = $basepath))"

echo "Reproducing figures and tables\n"
Rscript -e "rmarkdown::render('scripts/Multiomics_and_figures.Rmd',params=list(basepath = basepath = $basepath))"

echo "Done"
