import utils
import argparse
import pandas as pd
import time
import pathlib
from Bio import SeqIO

def main(task_id):

    """
    if end == 'all':
        gene_ids = list(nodes_df[nodes_df.node_type == 'gene/protein']['node_name'][int(start):])
    else:
        gene_ids = list(nodes_df[nodes_df.node_type == 'gene/protein']['node_name'])[int(start):int(end)]
    """
    
    start = time.time()
    model_name = 'esm2_t33_650M_UR50D'
    fasta_file = pathlib.Path(f'./gene_sequences{task_id}.fasta')
    output_dir = pathlib.Path(f'embeddings{task_id}')
    gene_dict_file = pathlib.Path(f'./gene_id_dict_{task_id}.csv')
    utils.extract_embeddings(model_name, fasta_file, output_dir, gene_dict_file)
    end = time.time()
    elapsed = end - start
    print(f"Fetching embeddings took {elapsed} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch gene sequences and write to a file.')
    parser.add_argument('-t', '--task_id')
    
    args = parser.parse_args()
    
    main(int(args.task_id))


    def remove_duplicates(fasta_file, output_file):
        unique_sequences = {}
        
        # Read the FASTA file and store unique sequences
        for record in SeqIO.parse(fasta_file, "fasta"):
            if str(record.seq) not in unique_sequences:
                unique_sequences[str(record.seq)] = record

        # Write unique sequences to a new file
        with open(output_file, "w") as output_handle:
            SeqIO.write(unique_sequences.values(), output_handle, "fasta")

    # Replace with your FASTA file path
    fasta_file = "gene_sequences_7.fasta"
    output_file = "gene_sequences7.fasta"

    #remove_duplicates(fasta_file, output_file)
