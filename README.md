# HMM_explore
This repository contains the code used to analyze the HMM model, as described in the paper 'Dissecting unsupervised learning through hidden Markov modelling in electrophysiological data', https://www.biorxiv.org/content/10.1101/2023.01.19.524547v2.

Please note that this repository uses the HMM toolbox implemented by Vidaurre et al, and publicly available at 'https://github.com/OHBA-analysis/HMM-MAR'. Make sure the toolbox is downloaded and added to the MATALAB path before you run the scripts. 

The repository contains:
- analysis_scripts folder, with the scripts used for the analyses described in the paper above mentioned.
- simdata folder, with the functions used to generate simulated data
- regression folder, with functions implementing cross-validated ridge regression

Please note that this repositorty does not include the real data used for the analyses.

! The LFP dataset used in 'test_HMM_LFP' is part of a yet unpublished dataset, but can be obtained upon request. 

! The MEG dataset used in 'test_HMM_MEG' has been previously collected by O'Neill et al. (2015), https://doi.org/10.1016/j.neuroimage.2015.04.030. This repository does not contain the data, neither the functions used to load them (can be obtained upon request).
