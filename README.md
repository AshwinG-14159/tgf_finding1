
This is an attempt at TGF Finding using the assumption that when czti faces the earth, there will be more millisecond binned outliers compared to when it is facing away from the earth due to tgfs.

However the results don't seem to suggest that this is the case.

To run these scripts, tgf_finding_final.py should be opened and appropriate paths to the orbitinfo.csv, the evt files and the mfk files should be given, along with the required binning, the number of orbits to analyse as well as the gap in the csv between 2 orbits analysed. Also give a version number for the table generation

It creates tables and they get stored in the specified directory. The nomenclature is mentioned.

Use tgf_finding_polar_final.py to do the same for polar orbits and create tables.

Then, to plot variations in outlier frequencies across extent of outlier and number of quadrants, use get_result_tables_final.py. Even in this, mention the parameters like gap, number and version from the original runs of table making.

Follow it up with a plot version number, used to keep track of different plots. 



Details



Runs currently - 


tgf_finding_final.py:
version 1, gap=10, runs = 100
version 2, gap=20, runs = 300
version 3, gap=30, runs = 400



tgf_finding_polar_final:
version 1, gap = 8, runs=100
version 2, gap = 14, runs = 300
version 3, gap = 13, runs = 400

plots - 




