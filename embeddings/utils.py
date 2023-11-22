import pathlib
import torch
from esm import FastaBatchedDataset, pretrained
from Bio import Entrez
import time
import pandas as pd
import re

# Always tell NCBI who you are (email)

def fetch_sequences(id_list, output_file, gene_id_dict, gene_id_filename, db='protein'):
    Entrez.email = "emma.tysinger@gmail.com"
    with open(output_file, 'w') as outfile:
        for i, gene_id in enumerate(id_list):
            # Fetch the sequence
            if i%100 == 0:
                print(f"Retrieved {i} of {len(id_list)} sequences")
            try:
                handle = Entrez.esearch(db=db, retmax=10, term=gene_id, idtype='acc')
                record = Entrez.read(handle)
                handle.close()
            except:
                pd.DataFrame(list(gene_id_dict.items()), columns=['gene_id', 'sequence']).to_csv(gene_id_filename, index=False)
                print(f"Failed on {i} of {len(id_list)} sequences")
                break

            gene_id_list = record["IdList"]
            try: 
                gene_id_dict[gene_id_list[0]] = gene_id
            except:
                print(f'No ids found for {gene_id}')
                continue
            
            try:
                handle = Entrez.efetch(db=db, id=gene_id_list[0], rettype="fasta", retmode="text")
            except:
                pd.DataFrame(list(gene_id_dict.items()), columns=['gene_id', 'sequence']).to_csv(gene_id_filename, index=False)
                print(f"Failed on {i} of {len(id_list)} sequences")
                break
            gene_data = handle.read()
            handle.close()
            
            # Write the sequence to a file
            outfile.write(gene_data)
            
            # NCBI recommends not to send more than 3 requests per second to avoid overload
            # on their servers, so we wait for a third of a second before the next request
            time.sleep(1)
    return gene_id_dict

def extract_embeddings(model_name, fasta_file, output_dir, gene_dict_file, tokens_per_batch=4096, seq_length=1022,repr_layers=[33]):
    df = pd.read_csv(gene_dict_file)
    gene_dict = pd.Series(df.sequence.values,index=df.gene_id).to_dict()

    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        
    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(tokens_per_batch, extra_toks_per_seq=1)

    data_loader = torch.utils.data.DataLoader(
        dataset, 
        collate_fn=alphabet.get_batch_converter(seq_length), 
        batch_sampler=batches
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):

            print(f'Processing batch {batch_idx + 1} of {len(batches)}')

            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)

            logits = out["logits"].to(device="cpu")
            representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}
            
            for i, label in enumerate(labels):
                entry_id = label.split()[0]
                try:
                    gene_id = gene_dict[entry_id]
                except:
                    entry_id = entry_id.split('|')[1]
                    try:
                        gene_id = gene_dict[entry_id]
                    except:
                        print(f"Can't get {label.split()[0]}")
                        continue
                
                filename = output_dir / f"{gene_id}.pt"
                truncate_len = min(seq_length, len(strs[i]))

                result = {"gene_id": gene_id, 
                          "entry_id": entry_id}
                result["mean_representations"] = {
                        layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                        for layer, t in representations.items()
                    }

                torch.save(result, filename)

def extract_protein_names(file_path):
    # Regular expression pattern to match the lines with protein names
    pattern = r"No ids found for (\w+)"

    # List to store extracted protein names
    protein_names = []

    # Open and read the file
    with open(file_path, 'r') as file:
        for line in file:
            # Use regular expression to find matches
            match = re.search(pattern, line)
            if match:
                # Append the found protein name to the list
                protein_names.append(match.group(1))

    return protein_names

