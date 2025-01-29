# DSC-180B
Team A06-1

Data source: <a href="https://string-db.org/cgi/download?sessionId=bLtv7nEpZD9a&species_text=Homo+sapiens&settings_expanded=0&min_download_score=0&filter_redundant_pairs=0&delimiter_type=txt">STRING Database</a>
Paper reference: <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10120954">PASNVGA</a>


### Repo File Documentation

- `Data` Folder: stores original and intermediate data files
    - `filtered_PPI.csv`: 
    - `PPI_protein_expression_full.csv`: full IHC protein expression feature matrix.
    - `PPI_protein_expression_10PCs.csv`: 10 principle components for the IHC protein expression feature matrix.
    - `PPI_RNA_seq_full.csv`: full GTEx RNA-seq feature matrix.
    - `PPI_RNA_seq_10PCs.csv`: 10 principle components for the GTEx RNA-seq feature matrix.
    - `protein_gene_conversion.csv`: conversion file between protein and gene IDs.
- `sample_data.py`: a script to sample a subset of data for experimenting and testing purpose.
