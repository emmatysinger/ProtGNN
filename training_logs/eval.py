import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse

def pretrain_plot(data, save_dir):
    df = pd.read_csv(data)
    metrics = ['Loss', 'AUROC', 'AUPRC']

    df['Epoch-Step'] = df.apply(lambda row: f"{int(row.Epoch)}-{int(row.Step)}", axis=1)
    sns.set_theme(context='notebook', style='white')
    sns.set_palette('colorblind')

    for metric in metrics:
        plt.figure(figsize=(12, 6))
        if metric == 'Loss':
            sns.lineplot(x=df['Epoch-Step'], y=df[metric], label=metric)
        else:
            sns.lineplot(x=df['Epoch-Step'], y=df['Micro '+metric], label='Micro '+metric)
            sns.lineplot(x=df['Epoch-Step'], y=df['Macro '+metric], label='Macro '+metric)
        plt.xlabel('Epoch-Step')
        plt.ylabel(metric)
        plt.title(f'{metric} during Pretraining')
        plt.xticks(rotation=45)  
        plt.legend()
        plt.tight_layout()  
        plt.savefig(os.path.join(save_dir, 'pretrain_'+metric+'.png'))
        plt.clf()

def finetune_plot(data, save_dir):
    df = pd.read_csv(data)

    metric_pairs = [
        ('Train Micro AUROC', 'Validation Micro AUROC'),
        ('Train Micro AUPRC', 'Validation Micro AUPRC'),
        ('Train Macro AUROC', 'Validation Macro AUROC'),
        ('Train Macro AUPRC', 'Validation Macro AUPRC'),
        ('Loss', 'Validation Loss')
    ]
    sns.set_theme(context='notebook', style='white')
    sns.set_palette('colorblind')  

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    axes = axes.flatten() 

    for idx, (train_metric, val_metric) in enumerate(metric_pairs[:-1]):
        ax = axes[idx]

        sns.lineplot(ax=ax, x=df['Epoch'], y=df[train_metric], label=f'{train_metric[:5]}')
        df[val_metric] = df[val_metric].interpolate()
        sns.lineplot(ax=ax, x=df['Epoch'], y=df[val_metric], label=f'{val_metric[:10]}')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Metric Value')
        ax.set_title(f'{train_metric[6:]}')
        ax.legend()
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'finetune_AUROC_AUPRC.png'))
    plt.clf()

    train_metric = 'Loss'
    val_metric  = 'Validation Loss'
    plt.figure(figsize=(12, 6))

    sns.lineplot(x=df['Epoch'], y=df[train_metric], label='Training '+train_metric, linestyle='-')

    df[val_metric] = df[val_metric].interpolate()
    sns.lineplot(x=df['Epoch'], y=df[val_metric], label=val_metric, linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{train_metric} and {val_metric} during Finetuning')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'finetune_loss.png'))
    plt.clf()

def main(args, log_type):
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    if log_type == 'pretrain':
        pretrain_plot(args.infile, args.savedir)
    elif log_type == 'finetune':
        finetune_plot(args.infile, args.savedir)
    else:
        raise ValueError('Please choose data type of either "pretrain" or "finetune"')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', required=True, help="Input log file")
    parser.add_argument('-s', '--savedir', required=True, help="Output CSV file")
    parser.add_argument('-t', '--type', required=True, help='Type of data file (pretrain or finetune)')
    args = parser.parse_args()

    main(args, args.type)
