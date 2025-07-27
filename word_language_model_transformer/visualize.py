#!/usr/bin/env python3
import argparse
import json

from matplotlib import pyplot as plt


def draw_loss(data, report_dir):
    epochs = [record['epoch'] for record in data['records']]
    val_losses = [record['val_loss'] for record in data['records']]
    ppl = [record['ppl'] for record in data['records']]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, val_losses, marker='o', label='Validation Loss')
    plt.title('Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, ppl, marker='o', color='orange', label='Perplexity')
    plt.title('Perplexity Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{report_dir}/loss_curves.png")


def main():
    parser = argparse.ArgumentParser(description='Visualize training reports')
    parser.add_argument('--report-dir', type=str, required=True,
                        help='Directory containing training report')
    args = parser.parse_args()

    report_fpath = f"{args.report_dir}/report.json"
    
    try:
        with open(report_fpath, 'r') as f:
            report_data = json.load(f)
            draw_loss(report_data, args.report_dir)
    except FileNotFoundError:
        print(f"No report found at {report_fpath}")



if __name__ == '__main__':
    main()
