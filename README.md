# IFNepitope2
A computational approach to predict, scan, and design the host-specific IFN-γ inducing epitopes using the sequence information of the peptides.
## Introduction
IFNepitope2 is an update of IFNepitope published by our group in 2013. It is developed to predict, scan, and, design the IFN-γ inducing peptides for human and mouse host, seperately, using sequence information only. In the standalone version, DPC based extra-tree classifier model is implemented alongwith the BLAST search, named it as hybrid approach.
IFNepitope2 is also available as web-server at https://webs.iiitd.edu.in/raghava/ifnepitope2. Please read/cite the content about the IFNepitope2 for complete information including algorithm behind the approach.

## PIP Installation
PIP version is also available for easy installation and usage of this tool. The following command is required to install the package 
```
pip install ifnepitope2
```
To know about the available option for the pip package, type the following command:
```
ifnepitope2 -h
```
## Standalone
The Standalone version of transfacpred is written in python3 and following libraries are necessary for the successful run:
- scikit-learn
- Pandas
- Numpy
- blastp

## Parameter Optimzation
For hyperparameter tuning, we implemented grid search with 5-fold stratified cross-validation across different classifiers, including Decision Trees (DT), Random Forest (RF), Logistic Regression (LR), XGBoost (XGB), K-Nearest Neighbors (KNN), Extra Trees (ET), and Support Vector Classifiers (SVC). Each model was optimized using a tailored parameter grid. The parameter grids were dynamically adjusted based on the data to ensure valid configurations during cross-validation. The best hyperparameters were selected based on the highest AUROC scores achieved during the tuning process. 

To know about the available option for the parameter optimization, type the following command:
```
python3 param_opt.py -h
```
To run the code with feature file, type the following command:
```
python3 param_opt.py --file <feature file> --Classifer <Classifier Options>
```

## Minimum USAGE
To know about the available option for the stanadlone, type the following command:
```
python ifnepitope2.py -h
```
To run the example, type the following command:
```
python3 ifnepitope2.py -i example_input_human.fa
```
This will predict if the submitted sequences are IFN-γ inducers or Non-inducer. It will use other parameters by default. It will save the output in "outfile.csv" in CSV (comma seperated variables).

## Full Usage
```
usage: ifnepitope2.py [-h]
		      -i INPUT
		      [-o OUTPUT]
		      [-s {1,2}]
		      [-j {1,2,3}]
                      [-t THRESHOLD]
                      [-w {8,9,10,11,12,13,14,15,16,17,18,19,20}]
		      [-d {1,2}]
```
```
Please provide following arguments

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input: protein or peptide sequence(s) in FASTA format
                        or single sequence per line in single letter code
  -o OUTPUT, --output OUTPUT
                        Output: File for saving results by default outfile.csv
  -s {1,2}, --host {1,2}
                        Host: 1: Human, 2: Mouse, by default 1
  -j {1,2,3}, --job {1,2,3}
                        Job Type: 1:Predict, 2: Design, 3:Scan, by default 1
  -t THRESHOLD, --threshold THRESHOLD
                        Threshold: Value between 0 to 1 by default 0.49
  -w {8,9,10,11,12,13,14,15,16,17,18,19,20}, --winleng {8,9,10,11,12,13,14,15,16,17,18,19,20}
                        Window Length: 8 to 20 (scan mode only), by default 8
  -d {1,2}, --display {1,2}
                        Display: 1:IFN-γ inducers, 2: All peptides, by default 1
```

**Input File:** It allow users to provide input in the FASTA format.

**Output File:** Program will save the results in the CSV format, in case user do not provide output file name, it will be stored in "outfile.csv".

**Threshold:** User should provide threshold between 0 and 1, by default its 0.49.

**Host:** User is allowed to choose the host organism, such as, 1 for Human, and 2 for Mouse.

**Job:** User is allowed to choose between three different modules, such as, 1 for prediction, 2 for Designing and 3 for scanning, by default its 1.

**Window length**: User can choose any pattern length between 8 and 20 in long sequences. This option is available for only scanning module.

**Display type:** This option allow users to fetch either only HLA-DRB1-04:01 binding peptides by choosing option 1 or prediction against all peptides by choosing option 2.

IFNepitope2 Package Files
=======================
It contantain following files, brief descript of these files given below

INSTALLATION                          : Installations instructions

LICENSE                               : License information

README.md                             : This file provide information about this package

model.zip                             : This zipped file contains the compressed version of models

envfile                               : This file compeises of paths for the database and blastp executable

ifnepitope2.py 	                      : Main python program

example_input_human.fa                : Example file contain peptide sequences for human host in FASTA format

example_input_mouse.fa                : Example file contain peptide sequenaces for mouse host in FASTA format

example_predict_human_output.csv      : Example output file for predict module for human host

example_predict_mouse_output.csv      : Example output file for predict module for mouse host

example_scan_human_output.csv         : Example output file for scan module for human host

example_scan_mouse_output.csv         : Example output file for scan module for mouse host

example_design_human_output.csv       : Example output file for design module for human host

example_design_mouse_output.csv       : Example output file for design module for mouse host
