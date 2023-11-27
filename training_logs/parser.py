import re
import csv
import argparse

def pretrain_parser(infile, outfile):
    # Regex to match the lines with metrics
    pattern = r'Epoch: (\d+) Step: (\d+) LR: (\S+) Loss (\S+), Pretrain Micro AUROC (\S+) Pretrain Micro AUPRC (\S+) Pretrain Macro AUROC (\S+) Pretrain Macro AUPRC (\S+)'

    # Open the log file and read line by line
    with open(infile, 'r') as file:
        lines = file.readlines()

    # Open the CSV file for writing
    with open(outfile, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header
        csvwriter.writerow(['Epoch', 'Step', 'Learning Rate', 'Loss', 'Micro AUROC', 'Micro AUPRC', 'Macro AUROC', 'Macro AUPRC'])

        # Iterate through each line in the log file
        for line in lines:
            match = re.search(pattern, line)
            if match:
                # Extract metrics
                epoch, step, lr, loss, micro_auroc, micro_auprc, macro_auroc, macro_auprc = match.groups()
                # Write the extracted metrics to the CSV file
                csvwriter.writerow([epoch, step, lr, loss, micro_auroc, micro_auprc, macro_auroc, macro_auprc])

    print(f"Metrics extracted and saved to {outfile}")

def finetune_parser(infile, outfile):
    pass


def main(args, log_type):
    if log_type == 'pretrain':
        pretrain_parser(args.infile, args.outfile)
    elif log_type == 'random':
        finetune_parser(args.infile, args.outfile)
    else:
        raise ValueError('Please choose log type of either "pretrain" or "finetune"')

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', required=True, help="Input log file")
    parser.add_argument('-o', '--outfile', required=True, help="Output CSV file")
    parser.add_argument('-t', '--type', required=True, help='Type of log file (pretrain or finetune)')
    args = parser.parse_args()

    main(args, args.type)

