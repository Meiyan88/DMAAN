# DMAAN for Imaging genetics

We provide the the code of Deep Multimodality-Disentangled Association Analysis Networ (DMAAN). 

## Data preprocess

For MRI data, region of interest (ROI) based features were extracted under the following preprocessing steps: (a) anterior commissure-posterior commissure correction with MIPAV software (https://mipav.cit.nih.gov/); (b) image intensity inhomogeneity correction by applying N3 algorithm; (c) skull stripping via HD-BET (https://github.com/MIC-DKFZ/HD-BET); (d) registering images to Montreal Neurological Institute (MNI) space via advanced normalization tools (ANTs) (https://github.com/ANTsX/ANTsPy); (e) three main tissues (i.e., gray matter (GM), white matter, and cerebrospinal fluid) segmentation using Atropos algorithm in ANTs; (f) applying the automated anatomical label (AAL) atlas of MNI space to project 90 brain ROIs; and (g) computing GM tissue volume of each ROI in the MNI space. For PET images, the co-registration strategy was utilized to align them to the corresponding MRI images via ANTs and the average intensity of each ROI was calculated as its feature. Finally, 90-dimensional ROI-based were separately extracted from the MRI and PET data for each subject.

For DTI data, 65 original format images where the b0 images do not activate the diffusion gradient were contained in each subject, whereas the other 64 images have different gradient directions. The DTI data were preprocessed by the following steps: (a) generating a b-vector file and a b-value file indicating each gradient direction and its scalar value via dcm2niix tool; (b) eddy correction with FMRIB Software Library (FSL); (c) skull stripping on b0 image using BET algorithm of FSL; (d) using generated files in step (a), the results of skull stripping and DTI images to calculate fractional anisotropy (FA) via difiti command in FSL; (e) aligning b0 image to MNI space via affine registration; (f) applying the transformation matrix in FA and labeling 90 ROIs with AAL; and (f) calculating the mean tissue density of each ROI of FA and then the corresponding 90-dimensional ROI-based features could be obtained.

For SNP data, the first line quality control steps include (a) call rate check per subject and SNP marker, (b) gender check, (c) sibling pair identification, (d) the Hardy-Weinberg equilibrium test, (e) marker removal by the minor allele frequency, and (f) population stratification. The second line preprocessing steps include the removal of SNPs with (a) more than 2% missing values, (b) minor allele frequencies of below 5\%, and (c) Hardy-Weinberg equilibrium 10e–6. The Michigan Imputation Server (https://imputationserver.readthedocs.io/en/latest/) with Minimac4 was applied on all subjects to perform genotype imputation, where 1000G Phase I Integrated Release Version 3 haplotypes (http://www.1000genomes.org)  was used as reference panel. Additionally, a global sure independence screening procedure presented in our previous study was applied to select the candidate SNPs. Herein, the selection of p-values in the SNP screening procedure is based on the amount of data in different datasets. 

## More information
For more information about PKAFnet, please read the following paper:

    Tao Wang,  Xiumei Chen, Jiawei Zhang, Qianjin Feng *, Meiyan Huang *. Deep Mult-imodality-disentangled Association Analysis Network for Imaging Genetics in Neurodegenerative Diseases, Medical Image Analysis, 2023. 

      
Please also cite this paper if you are using DMAAN for your research!
