# PPI-OMEGA

This is the project repository for Data Science Capstone at UCSD in 2025 by Team A06-1.


## Installation and Execution Guide

### **1. Clone the Repository**
```bash
git clone https://github.com/EliteApex/PPI-OMEGA.git
cd PPI-OMEGA
```

### **2. Environment setup**
```bash
# After ensuring conda is installed ...
conda env create -f environment.yml
conda activate PPIOMEGA_env  
```

### **3. Running the Project**

Once the environment is set up, you can run the project using:

```bash
python src/main.py 
```


## Project Structure

The repository is organized as follows:

```text
.
├── src/                # Source code for the project
│   ├── preprocessing.py      # script to preprocess the data and create the input feature matrix 
│   ├── model.py      # script of essential components of the model
│   └── main.py         # Entry point for running the project
├── environment.yml     # Conda environment file
├── README.md           # Documentation
└── Data/               # Directory for dataset, stores original and intermediate data files
   ├── filtered_PPI.csv      # Top 5% scored PPI dataset
   ├── PPI_protein_expression_full.csv     # full IHC protein expression feature matrix
   ├── PPI_protein_expression_10PCs.csv     # 10 principle components for the protein expression feature matrix
   ├── PPI_RNA_seq_full.csv     # full GTEx RNA-seq feature matrix
   ├── PPI_RNA_seq_10PCs.csv     # 10 principle components for the GTEx RNA-seq feature matrix
   └── protein_gene_conversion.csv         # conversion file between protein and gene IDs
```

<!-- `sample_data.py`: a script to sample a subset of data for experimenting and testing purpose. -->

<!-- Data source: <a href="https://string-db.org/cgi/download?sessionId=bLtv7nEpZD9a&species_text=Homo+sapiens&settings_expanded=0&min_download_score=0&filter_redundant_pairs=0&delimiter_type=txt">STRING Database</a>
Paper reference: <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10120954">PASNVGA</a> -->