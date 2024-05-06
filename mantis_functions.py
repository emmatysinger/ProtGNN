import pandas as pd
import requests
import json

def get_uniprot_function(gene_name):
    """Fetches UniProt function description for a given gene name."""
    url = f"https://rest.uniprot.org/uniprotkb/search?query=(reviewed:true)%20AND%20(organism_id:9606)%20AND%20(gene:{gene_name})"
    response = requests.get(url)
    if response.ok:
        data = json.loads(response.text)
        if data["results"]:
            try:
                function_description = [c for c in data['results'][0]['comments'] if c['commentType'] in ['FUNCTION', 'MISCELLANEOUS']][0]['texts'][0]['value']
                return function_description
            except:
                print(f'{gene_name} have no function comment')
                return None
        else:
            #print(f'{gene_name} not found in UniProt')
            return None
    else:
        return "Error in fetching data"

mapping_file = pd.read_csv('TxGNN/txgnn_idx_name.csv')
mapping_file = mapping_file.sort_values(by='idx')
mapping_file = mapping_file.reset_index(drop=True)

results = []
for i, gene in enumerate(list(mapping_file['name'])):
    function_description = get_uniprot_function(gene)
    results.append({'Gene': gene, 'Function': function_description})
    if i%500 == 0:
        results_df = pd.DataFrame(results)
        results_df.to_csv('TxGNN/gene_to_function.csv')
        print(f'Functions fetched for {i} genes so far!')

results_df = pd.DataFrame(results)
results_df.to_csv('TxGNN/gene_to_function.csv')

joined_df = pd.merge(results_df, mapping_file, left_on='Gene', right_on='name')
joined_df.to_csv('TxGNN/gene_to_function_map.csv')

