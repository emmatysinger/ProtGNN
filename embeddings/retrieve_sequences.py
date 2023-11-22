import utils
import argparse
import pandas as pd
import time

def main(task_id, count, output_filename,gene_id_filename):
    nodes_df = pd.read_csv('../../kg/node.csv', sep='\t')
    gene_ids = list(nodes_df[nodes_df.node_type == 'gene/protein']['node_name'])
    gene_ids = gene_ids[task_id-1:len(gene_ids):count]
    # Replace 'missing_retrieval.txt' with the path to your text file
    file_path = 'missing_retrieval.txt'
    gene_ids = utils.extract_protein_names(file_path)

    """
    if end == 'all':
        gene_ids = list(nodes_df[nodes_df.node_type == 'gene/protein']['node_name'][int(start):])
    else:
        gene_ids = list(nodes_df[nodes_df.node_type == 'gene/protein']['node_name'])[int(start):int(end)]
    """
    
    gene_id_dict = {}
    start = time.time()
    gene_id_dict = utils.fetch_sequences(gene_ids, output_filename, gene_id_dict, gene_id_filename, db='nuccore')
    end = time.time()
    elapsed = end - start
    print(f"Fetching sequences for {len(gene_ids)} proteins took {elapsed} seconds.")
    pd.DataFrame(list(gene_id_dict.items()), columns=['gene_id', 'sequence']).to_csv(gene_id_filename, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch gene sequences and write to a file.')
    parser.add_argument('-t', '--task_id')
    parser.add_argument('-c', '--task_count')
    parser.add_argument('-o', '--output_filename', default='gene_sequences.fasta', 
                        help='Output filename for the gene sequences')
    parser.add_argument('-i', '--id_dict_filename', default='gene_id_dict.csv', 
                        help='Output filename the gene to id dictionary')
    
    args = parser.parse_args()
    
    main(int(args.task_id), int(args.task_count), args.output_filename, args.id_dict_filename)