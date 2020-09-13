
FROM andreanuzzo/mmim:0.2

CMD Rscript -e "installed.packages()" > /home/rstudio/MMIM/meh.tsv

CMD python3 -m venv /home/rstudio/.venv && \
	. /home/rstudio/.venv/bin/activate && \
	pip3 install numpy==1.18.1 \
		matplotlib==3.1.1 \
		pandas==0.25.3 \
		wheel && \
	pip3 install -r home/rstudio/MMIM/requirements.txt && \
	deactivate

CMD . /home/rstudio/.venv/bin/activate && which pip3 && pip3 list > /home/rstudio/MMIM/meh2.tsv
