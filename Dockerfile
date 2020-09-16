
# load base image
FROM rocker/rstudio:3.6.3

# Debian libraries

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update

RUN apt-get install -y python3.6 python3-pip python3-venv libxml2-dev

WORKDIR /home/rstudio



## packages

ENV VIRTUAL_ENV=/home/rstudio/.venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ENV RENV_VERSION=0.12.0 

COPY --chown=rstudio:rstudio renv.lock /home/rstudio/renv.lock
COPY --chown=rstudio:rstudio requirements.txt /home/rstudio/requirements.txt


# Install dependencies:
RUN pip3 install \
	numpy==1.18.1 \
	matplotlib==3.1.1 \
	pandas==0.25.3 \
	wheel && \
	pip3 install -r requirements.txt

RUN R -e "install.packages('remotes', repos = c(CRAN = 'https://cloud.r-project.org'))"
RUN R -e "remotes::install_github('rstudio/renv@${RENV_VERSION}')"
RUN R -e "renv::restore('/home/rstudio/')"

ENV PATH_TO_YOUR_DIRECTORY=/home/rstudio/MMIM

CMD #sh /home/rstudio/MMIM/scripts/get_data.sh $PATH_TO_YOUR_DIRECTORY && \ 
CMD sh /home/rstudio/MMIM/scripts/run_analysis.sh $PATH_TO_YOUR_DIRECTORY