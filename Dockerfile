FROM andreanuzzo/mmim:0.3

ENV PATH_TO_YOUR_DIRECTORY=/home/rstudio/MMIM

CMD chmod 700 /home/rstudio/MMIM/scripts/get_data.sh

CMD sh /home/rstudio/MMIM/scripts/get_data.sh /home/rstudio/MMIM

CMD sh /home/rstudio/MMIM/scripts/run_analysis.sh $PATH_TO_YOUR_DIRECTORY