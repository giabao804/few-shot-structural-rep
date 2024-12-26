#!/usr/bin/env python3
# train_5shot.py

import argparse
import os
import time
import logging

import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from CWRU.CWRU_dataset import CWRU
from dataloader.dataloader import FewshotDataset
from function.function import (
    ContrastiveLoss,
    cal_accuracy_fewshot_5shot,
    cal_metrics_fewshot_5shot,
    predicted_fewshot_5shot,
    seed_func,
    convert_for_5shots,
)
import function.function as function
from net.new_proposed import MainNet
from sklearn.manifold import TSNE


def setup_logging(log_file: str = "training.log") -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Bearing Faults Diagnosis')

    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--h', type=int, default=16, help='Height of the input image')
    parser.add_argument('--w', type=int, default=16, help='Width of the input image')
    parser.add_argument('--c', type=int, default=64, help='Number of channels of the input image')
    parser.add_argument('--dataset', choices=['CWRU'], required=True, help='Choose dataset for training')
    parser.add_argument('--training_samples_CWRU', type=int, default=60, help='Number of training samples for CWRU')
    parser.add_argument('--model_name', type=str, default='MainNet', help='Model name')
    parser.add_argument('--episode_num_train', type=int, default=130, help='Number of training episodes')
    parser.add_argument('--episode_num_test', type=int, default=150, help='Number of testing episodes')
    parser.add_argument('--way_num_CWRU', type=int, default=10, help='Number of classes for CWRU')
    parser.add_argument('--noise_DB', type=float, default=None, help='Noise database')
    parser.add_argument('--spectrum', action='store_true', help='Use spectrum')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--step_size', type=int, default=10, help='Step size for learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for learning rate scheduler')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--loss1', default=ContrastiveLoss(), help='Primary loss function')
    parser.add_argument('--loss2', default=nn.CrossEntropyLoss(), help='Secondary loss function')
    parser.add_argument('--path_weights', type=str, default='checkpoints/', help='Path to save weights')
    parser.add_argument('--data_path', type=str, default="../few-shot-structural-rep/CWRU/", help="Data path")
    parser.add_argument('--cfs_matrix', action='store_false', help="Print confusion matrix")
    parser.add_argument('--train_mode', action='store_false', help="Select train mode")
    parser.add_argument('--shot_num', type=int, default=5, help='Number of samples per class')
    args = parser.parse_args()
    return args


def load_dataset(args: argparse.Namespace) -> tuple:
    if args.dataset == 'CWRU':
        window_size = 2048
        split = args.training_samples_CWRU // 30
        data = CWRU(split, ['12DriveEndFault'], ['1772', '1750', '1730'], window_size)
        data.nclasses, data.classes, len(data.X_train), len(data.X_test)
        data.X_train = data.X_train.astype(np.float32)
        data.X_test = data.X_test.astype(np.float32)
        train_data_CWRU = torch.from_numpy(data.X_train).reshape([args.training_samples_CWRU, 4096])
        train_label_CWRU = torch.from_numpy(data.y_train)
        test_data_CWRU = torch.from_numpy(data.X_test).reshape([750, 4096])
        test_label_CWRU = torch.from_numpy(data.y_test)

        if args.noise_DB is not None:
            snr_dB = args.noise_DB
            data.add_noise_to_test_data(snr_dB, 0.001)
            noisy_test_data = data.X_test_noisy.reshape([750, 4096])

            if args.spectrum:
                train_data_CWRU = function.to_spectrum(train_data_CWRU)
                test_data_CWRU = function.to_spectrum(noisy_test_data)
            else:
                train_data_CWRU = train_data_CWRU.reshape(train_data_CWRU.shape[0], 1, 64, 64)
                test_data_CWRU = test_data_CWRU.reshape(test_data_CWRU.shape[0], 1, 64, 64)
        else:
            if args.spectrum:
                train_data_CWRU = function.to_spectrum(train_data_CWRU)
                test_data_CWRU = function.to_spectrum(test_data_CWRU)
            else:
                train_data_CWRU = train_data_CWRU.reshape(train_data_CWRU.shape[0], 1, 64, 64)
                test_data_CWRU = test_data_CWRU.reshape(test_data_CWRU.shape[0], 1, 64, 64)

        print('Shape of CWRU train data:', train_data_CWRU.shape)
        print('Shape of CWRU test data:', test_data_CWRU.shape)
        print('End Loading CWRU')

        train_dataset_CWRU = FewshotDataset(train_data_CWRU, train_label_CWRU, episode_num=args.episode_num_train, way_num=args.way_num_CWRU, shot_num=args.shot_num, query_num=1)
        train_dataloader_CWRU = DataLoader(train_dataset_CWRU, batch_size=args.batch_size, shuffle=True)
        test_dataset_CWRU = FewshotDataset(test_data_CWRU, test_label_CWRU, episode_num=args.episode_num_test, way_num=args.way_num_CWRU, shot_num=args.shot_num, query_num=1)
        test_dataloader_CWRU = DataLoader(test_dataset_CWRU, batch_size=args.batch_size, shuffle=False)

        return train_dataloader_CWRU, test_dataloader_CWRU, data.nclasses
    else:
        raise ValueError(f'Unsupported dataset: {args.dataset}')

def train_and_test_model(
    net: nn.Module,
    train_dataloader: DataLoader,
    test_loader: DataLoader,
    training_samples: int,
    num_epochs: int,
    lr: float,
    loss1: nn.Module,
    path_weight: str,
    num_samples: int,
    num_classes: int,
    model_name: str,
    device: torch.device
) -> tuple:
    
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    loss.to(device)

    full_loss = []
    full_metric = {'full_acc': [], 'full_f1': [], 'full_recall': []}
    pred_metric = {'pred_acc': 0, 'pred_f1': 0, 'pred_recall': 0}

    cumulative_time = 0
    best_model_path = None
    model_name = ''

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        running_loss = 0
        num_batches = 0
        optimizer.zero_grad()
        print('='*50, 'Epoch:', epoch, '='*50)
        net.train()

        with tqdm(train_dataloader, desc=f'Epoch {epoch}/{num_epochs}', unit='batch') as t:
            for query_images, query_targets, support_images, support_targets in t:
                q = query_images.permute(1, 0, 2, 3, 4).to(device)
                s = convert_for_5shots(support_images, support_targets, device)
                targets = query_targets.to(device).permute(1, 0).long()
                for i in range(len(q)):
                    out, _, _ = net(q[i], s)
                    target = targets[i]
                    loss = loss(out, target)
                    loss.backward()
                    running_loss += loss.detach().item()
                    num_batches += 1
                optimizer.step()
                optimizer.zero_grad()
                t.set_postfix(loss=running_loss / num_batches)

        elapsed_time = time.time() - start_time
        cumulative_time += elapsed_time
        cumulative_minutes = cumulative_time / 60
        print(f"Epoch {epoch}/{num_epochs} completed in {cumulative_minutes:.2f} minutes")

        scheduler.step()

        with torch.no_grad():
            total_loss = running_loss / num_batches
            full_loss.append(total_loss)
            print('------------Testing on the test set-------------')
            acc, f1, recall = cal_metrics_fewshot_5shot(test_loader, net, device, num_classes)
            full_metric['full_acc'].append(acc)
            full_metric['full_f1'].append(f1)
            full_metric['full_recall'].append(recall)
            print(f'Accuracy on the test set: {acc:.4f}')
            print(f'F1_score on the test set: {f1:.4f}')
            print(f'Recall on the test set: {recall:.4f}')
            if acc > pred_metric['pred_acc']:
                if epoch >= 2 and best_model_path:
                    if os.path.exists(best_model_path):
                        os.remove(best_model_path)
                pred_metric['pred_acc'] = acc
                pred_metric['pred_f1'] = f1
                pred_metric['pred_recall'] = recall
                model_name = f'{args.model_name}_5shot_recall_{recall:.4f}_{training_samples}samples.pth'
                best_model_path = os.path.join(path_weight, model_name)
                torch.save(net.state_dict(), best_model_path)
                print(f'=> Save the best model with accuracy: {acc:.4f}')
    torch.cuda.empty_cache()
    return full_loss, full_metric, model_name, pred_metric['pred_acc'], pred_metric['pred_f1'], pred_metric['pred_recall']


def main():
    setup_logging()
    args = parse_arguments()
    print(args)

    seed_func()
    print('Seed set for reproducibility.')

    train_dataloader, test_dataloader, num_classes = load_dataset(args)

    net = MainNet()
    net = net.to(args.device)

    os.makedirs(args.path_weights, exist_ok=True)

    if args.train_mode:
        print('Starting training phase...')
        _, _, model_name, acc, _, _ = train_and_test_model(
            net=net,
            train_dataloader=train_dataloader,
            test_loader=test_dataloader,
            training_samples=args.training_samples_CWRU,
            num_epochs=args.num_epochs,
            lr=args.lr,
            loss=args.loss1,
            path_weight=args.path_weights,
            num_samples=args.training_samples_CWRU,
            num_classes=args.way_num_CWRU
        )
        print('End training...................!!')

    if args.cfs_matrix:
        print("Validating...")
        faults_idx = {
            'Normal': 0,
            '0.007-Ball': 1,
            '0.014-Ball': 2,
            '0.021-Ball': 3,
            '0.007-Inner': 4,
            '0.014-Inner': 5,
            '0.021-Inner': 6,
            '0.007-Outer': 7,
            '0.014-Outer': 8,
            '0.021-Outer': 9,
        }

        net = MainNet()
        saved_weights_path = os.path.join(args.path_weights, model_name)
        net.load_state_dict(torch.load(saved_weights_path))
        net = net.to(args.device)
        net.eval()

        true_labels, predicted, _, _ = predicted_fewshot_5shot(test_dataloader, net, args.device)

        faults_labels = {v: k for k, v in faults_idx.items()}
        unique_labels = np.unique(true_labels)
        tick_labels = [faults_labels[label] for label in unique_labels]
        print(tick_labels)

        predicted = predicted.squeeze()
        predicted_labels = np.argmax(predicted, axis=1)
        confusion = confusion_matrix(true_labels.squeeze(), predicted_labels)
        plt.figure(figsize=(10, 8))
        plt.imshow(confusion, cmap='RdPu')
        plt.colorbar()
        plt.xlabel('Predicted Labels', fontsize=16)
        plt.ylabel('Actual Labels', fontsize=16)
        plt.title('Confusion Matrix', fontsize=16)

        total = np.sum(confusion) / args.way_num_CWRU
        save_path = f"{args.path_weights}cfs_{args.training_samples_CWRU}_{acc}.png"

        for i in range(confusion.shape[0]):
            for j in range(confusion.shape[1]):
                count = confusion[i, j]
                percent = (count / total) * 100
                text_color = 'white' if count > 50 else 'black'
                plt.text(j, i - 0.1, f'{count}', ha='center', va='center', color=text_color, fontsize=11)
                plt.text(j, i + 0.2, f'({percent:.1f}%)', ha='center', va='center', color=text_color, fontsize=9)

        tick_locations = np.arange(len(unique_labels))
        plt.xticks(tick_locations, tick_labels, rotation=45, ha='right', fontsize=9)
        plt.yticks(tick_locations, tick_labels, rotation=45, ha='right', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.show()

        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(predicted)
        plt.figure(figsize=(5, 5))
        plt.grid(True, ls='--', alpha=0.5)
        unique_labels = np.unique(true_labels)
        num_classes = len(unique_labels)
        color_map = plt.cm.get_cmap('Paired', num_classes)
        for i, label in enumerate(unique_labels):
            class_indices = np.where(true_labels == label)
            plt.scatter(
                tsne_results[class_indices, 0],
                tsne_results[class_indices, 1],
                label=f'Class {label}',
                color=color_map(i),
                s=30,
                alpha=0.8,
                linewidths=2
            )
        plt.tight_layout()
        plt.legend()
        tsne_save_path = f"{args.path_weights}tsne_{args.training_samples_CWRU}_{acc}.png"
        plt.savefig(tsne_save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.show()


if __name__ == '__main__':
    main()
