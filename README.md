# ComputerScience
Repository for the paper written for the Computer Science for Business Analytics course
This project is about creating a scalable duplicate detection method for web shop products.
First, the titles are cleaned, then we apply LSH on the signatures based on model words. 
The LSH candidate pairs are then clustered using average linkage hierarchical clustering, to predict duplicates. 

Structure:
The code contains many self-explaining functions: create_distancematrix creates a distance matrix, etc. 
create_dataframe(data) on line 406 creates the (cleaned) dataframe that will be used by the rest of the program. bootstrap(tvDF, nr_bootstraps, bootstrap_perc, dist_threshold_list, br_pairs) calls all the appropriate functions in the right order with the cleaned data, for the desired amount of bootstraps, percentage of training data, distance thresholds, and br pairs.

How to use:
Change the path at the top to where you have stored TVs-all-merged.json
Optionally:
Change br_pairs to the desired pairs, the current vector is how it has been run for the paper. 
Note that b and r have to be integers, and the program computes the signature length n as b*r. 
Change dist_threshold_list to the distance thresholds you desire.
Change nr_bootstraps to the desired amount of bootstraps
Change the bootstrap_perc to the desired percentage.
