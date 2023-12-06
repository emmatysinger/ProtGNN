import re
import csv
import argparse
import pandas as pd

def pretrain_parser(infile, outfile):
    pattern = r'Epoch: (\d+) Step: (\d+) LR: (\S+) Loss (\S+), Pretrain Micro AUROC (\S+) Pretrain Micro AUPRC (\S+) Pretrain Macro AUROC (\S+) Pretrain Macro AUPRC (\S+)'

    with open(infile, 'r') as file:
        lines = file.readlines()

    with open(outfile, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Epoch', 'Step', 'Learning Rate', 'Loss', 'Micro AUROC', 'Micro AUPRC', 'Macro AUROC', 'Macro AUPRC'])

        for line in lines:
            match = re.search(pattern, line)
            if match:
                epoch, step, lr, loss, micro_auroc, micro_auprc, macro_auroc, macro_auprc = match.groups()
                csvwriter.writerow([epoch, step, lr, loss, micro_auroc, micro_auprc, macro_auroc, macro_auprc])

    print(f"Metrics extracted and saved to {outfile}")

def finetune_parser(infile, outfile):
    with open(infile, 'r') as file:
        lines = file.readlines()

    epochs_data = []
    current_epoch_data = {}
    current_epoch = -1

    for line in lines:
        if "Epoch:" in line:
            epoch_parts = line.split(',')
            epoch = int(re.search(r'Epoch: (\d+)', epoch_parts[0]).group(1))
            if epoch > current_epoch:
                # Ensure all keys exist even if validation data is missing
                for key in ['Validation Loss', 'Validation Micro AUROC', 'Validation Micro AUPRC', 'Validation Macro AUROC', 'Validation Macro AUPRC']:
                    current_epoch_data.setdefault(key, None)
                epochs_data.append(current_epoch_data)
                current_epoch_data = {}
                current_epoch = epoch
            
            current_epoch_data['Epoch'] = int(re.search(r'Epoch: (\d+)', epoch_parts[0]).group(1))
            current_epoch_data['LR'] = float(re.search(r'LR: (\d+.\d+)', epoch_parts[0]).group(1))

            if 'Train' in line:
                current_epoch_data['Loss'] = float(re.search(r'Loss (\d+.\d+)', epoch_parts[0]).group(1))
                current_epoch_data['Train Micro AUROC'] = float(re.search(r'Train Micro AUROC (\d+.\d+)', epoch_parts[1]).group(1))
                current_epoch_data['Train Micro AUPRC'] = float(re.search(r'Train Micro AUPRC (\d+.\d+)', epoch_parts[1]).group(1))
                current_epoch_data['Train Macro AUROC'] = float(re.search(r'Train Macro AUROC (\d+.\d+)', epoch_parts[1]).group(1))
                current_epoch_data['Train Macro AUPRC'] = float(re.search(r'Train Macro AUPRC (\d+.\d+)', epoch_parts[1]).group(1))

            elif 'Validation' in line:
                current_epoch_data['Validation Loss'] = float(re.search(r'Validation Loss (\d+.\d+)', epoch_parts[0]).group(1))
                current_epoch_data['Validation Micro AUROC'] = float(re.search(r'Validation Micro AUROC (\d+.\d+)', epoch_parts[1]).group(1))
                current_epoch_data['Validation Micro AUPRC'] = float(re.search(r'Validation Micro AUPRC (\d+.\d+)', epoch_parts[1]).group(1))
                current_epoch_data['Validation Macro AUROC'] = float(re.search(r'Validation Macro AUROC (\d+.\d+)', epoch_parts[1]).group(1))
                current_epoch_data['Validation Macro AUPRC'] = float(re.search(r'Validation Macro AUPRC (\d+.\d+)', epoch_parts[1]).group(1))

    # Add the last epoch data
    if current_epoch_data:
        # Ensure all keys exist even if validation data is missing
        for key in ['Validation Loss', 'Validation Micro AUROC', 'Validation Micro AUPRC', 'Validation Macro AUROC', 'Validation Macro AUPRC']:
            current_epoch_data.setdefault(key, None)
        epochs_data.append(current_epoch_data)

    # Writing to CSV
    headers = ['Epoch', 'LR', 'Loss', 'Train Micro AUROC', 'Train Micro AUPRC', 'Train Macro AUROC', 'Train Macro AUPRC', 'Validation Loss', 'Validation Micro AUROC', 'Validation Micro AUPRC', 'Validation Macro AUROC', 'Validation Macro AUPRC']
    with open(outfile, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for data in epochs_data:
            writer.writerow(data)


def main(args, log_type):
    if log_type == 'pretrain':
        pretrain_parser(args.infile, args.outfile)
    elif log_type == 'finetune':
        finetune_parser(args.infile, args.outfile)
    else:
        raise ValueError('Please choose log type of either "pretrain" or "finetune"')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', required=True, help="Input log file")
    parser.add_argument('-o', '--outfile', required=True, help="Output CSV file")
    parser.add_argument('-t', '--type', required=True, help='Type of log file (pretrain or finetune)')
    args = parser.parse_args()

    main(args, args.type)

