# load base image
FROM rocker/rstudio:3.6.2

# Debian libraries

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update

RUN apt-get install -y python3.6 python3-pip python3-venv libxml2-dev r-cran-devtools

## packages

COPY --chown=rstudio:rstudio renv.lock /home/rstudio/renv.lock
COPY --chown=rstudio:rstudio requirements.txt /home/rstudio/requirements.txt

ENV RENV_VERSION=0.12.0 

RUN R -e "install.packages('remotes', repos = c(CRAN = 'https://cloud.r-project.org'))"
RUN R -e "remotes::install_github('rstudio/renv@${RENV_VERSION}')"
RUN R -e "renv::restore('/home/rstudio/')"


RUN chmod 700 /home/rstudio/MMIM/scripts/get_data.sh

CMD bash /home/rstudio/MMIM/scripts/get_data.sh /home/rstudio/MMIM