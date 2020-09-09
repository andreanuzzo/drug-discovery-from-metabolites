# load base image
FROM rocker/rstudio:3.6.2

# Debian libraries
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y python3.6 python3-pip python3-venv libxml2-dev r-cran-devtools

## R packages
ARG RENV_VERSION=0.12.0 
RUN  R -e "install.packages('remotes', repos = c(CRAN = 'https://cloud.r-project.org'))" && \
  R -e "remotes::install_github('rstudio/renv@${RENV_VERSION}')" && \
  R -e 'renv::restore()'

## Python environment (requires venv)
COPY requirements.txt /tmp/requirements.txt
RUN /bin/bash -c "python3 -m venv .venv && \
  source .venv/bin/activate && \
  yes | pip3 install wheel numpy==1.18.1 matplotlib==3.1.1 pandas==0.25.3 && \
  yes | pip3 install -r /tmp/requirements.txt && \
  deactivate"
COPY .venv 
  
