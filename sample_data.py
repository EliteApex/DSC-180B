import pandas as pd
import random

# Note that this dataset is too big to be put on GitHub. You can find the file at https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz 
data = pd.read_csv('../9606.protein.links.v12.0.txt', sep='\t', names=['link_info'])
data = data.loc[1:]
data[['protein1', 'protein2', 'combined_score']] = data['link_info'].str.split(expand=True)
data.drop(columns=['link_info'], inplace=True)
data['combined_score'] = data['combined_score'].astype(int)
data['protein1'] = data['protein1'].apply(lambda x: x.replace("9606.",""))
data['protein2'] = data['protein2'].apply(lambda x: x.replace("9606.",""))
data.to_csv("../cleaned_PPI_full_name.csv")

random.seed(42)

all_proteins = pd.concat([data['protein1'], data['protein2']]).unique()
sampled_proteins = set(random.sample(list(all_proteins), 1000))

filtered_data = data[
    data['protein1'].isin(sampled_proteins) & data['protein2'].isin(sampled_proteins)
]
filtered_data.to_csv('filtered_PPI.csv', index=False)

print(f"Filtered data has {len(filtered_data)} links involving the sampled proteins.") #32558