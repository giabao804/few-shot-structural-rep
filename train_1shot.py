#!/usr/bin/env python3
# train_1shot.py

import argparse
import os
import time
import logging


import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from CWRU.CWRU_dataset import CWRU
from dataloader.dataloader import FewshotDataset
from function.function import (
    ContrastiveLoss,
    cal_accuracy_fewshot,
    cal_metrics_fewshot,
    predicted_fewshot,
    seed_func,
)
import function.function as function
# from net.new_proposed import MainNet


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
    parser = argparse.ArgumentParser(description='Bearing Faults Project Configuration')

    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')

    parser.add_argument('--h', type=int, default=16, help='Height of the input image')
    parser.add_argument('--w', type=int, default=16, help='Width of the input image')
    parser.add_argument('--c', type=int, default=64, help='Number of channels of the input image')
    parser.add_argument('--dataset', choices=['CWRU'], required=True, help='Choose dataset for training')
    parser.add_argument('--training_samples_CWRU', type=int, default=30, help='Number of training samples for CWRU')
    parser.add_argument('--way_num_CWRU', type=int, default=10, help='Number of classes for CWRU')
    parser.add_argument('--model_name', type=str, default='MainNet', help='Model name')
    parser.add_argument('--episode_num_train', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--episode_num_test', type=int, default=75, help='Number of testing episodes')
    parser.add_argument('--noise_DB', type=float, default=None, help='Noise database (SNR in dB)')
    parser.add_argument('--spectrum', action='store_true', help='Use spectrum representation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--step_size', type=int, default=10, help='Step size for learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for learning rate scheduler')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--loss', type=str, default='ContrastiveLoss', help='Loss function name')
    parser.add_argument('--path_weights', type=str, default='checkpoints/', help='Path to save weights')
    parser.add_argument('--data_path', type=str, default="../few-shot-structural-rep/CWRU/", help="Data path")
    parser.add_argument('--cfs_matrix', action='store_true', help="Print confusion matrix")
    parser.add_argument('--train_mode', action='store_true', help="Select train mode")
    parser.add_argument('--shot_num', type=int, default=1, help='Number of samples per class')

    return parser.parse_args()

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
    loss1.to(device)

    full_loss = []
    full_metric = {'full_acc': [], 'full_f1': [], 'full_recall': []}
    pred_metric = {'pred_acc': 0, 'pred_f1': 0, 'pred_recall': 0}
    cumulative_time = 0
    best_model_path = None

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        running_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        logging.info('=' * 50 + f' Epoch: {epoch} ' + '=' * 50)
        net.train()

        with tqdm(train_dataloader, desc=f'Epoch {epoch}/{num_epochs}', unit='batch') as t:
            for batch_idx, (query_images, query_targets, support_images, support_targets) in enumerate(t, 1):
                q = query_images.permute(1, 0, 2, 3, 4).to(device)
                s = support_images.permute(1, 0, 2, 3, 4).to(device)
                targets = query_targets.to(device).permute(1, 0).long()

                for i in range(len(q)):
                    out, _, _ = net(q[i], s)
                    target = targets[i]
                    loss = loss1(out, target)
                    loss.backward()
                    running_loss += loss.detach().item()
                    num_batches += 1

                optimizer.step()
                optimizer.zero_grad()
                current_loss = running_loss / num_batches
                t.set_postfix(loss=current_loss)

        elapsed_time = time.time() - start_time
        cumulative_time += elapsed_time
        cumulative_minutes = cumulative_time / 60
        logging.info(f"Epoch {epoch}/{num_epochs} completed in {cumulative_minutes:.2f} minutes")

        scheduler.step()

    
        net.eval()
        with torch.no_grad():
            total_loss = running_loss / num_batches
            full_loss.append(total_loss)
            logging.info('------------ Testing on the test set -------------')
            acc, f1, recall = cal_metrics_fewshot(test_loader, net, device, num_classes)
            full_metric['full_acc'].append(acc)
            full_metric['full_f1'].append(f1)
            full_metric['full_recall'].append(recall)

            logging.info(f'Accuracy on the test set: {acc:.4f}')
            logging.info(f'F1_score on the test set: {f1:.4f}')
            logging.info(f'Recall on the test set: {recall:.4f}')

            # Save the best model based on accuracy
            if acc > pred_metric['pred_acc']:
                if epoch >= 2 and best_model_path:
                    if os.path.exists(best_model_path):
                        os.remove(best_model_path)
                pred_metric['pred_acc'] = acc
                pred_metric['pred_f1'] = f1
                pred_metric['pred_recall'] = recall
                model_name_updated = f'{model_name}_1shot_recall_{recall:.4f}_{training_samples}samples.pth'
                best_model_path = os.path.join(path_weight, model_name_updated)
                torch.save(net.state_dict(), best_model_path)
                logging.info(f'=> Save the best model with accuracy: {acc:.4f}')

        torch.cuda.empty_cache()

    return full_loss, full_metric, os.path.basename(best_model_path), pred_metric['pred_acc'], pred_metric['pred_f1'], pred_metric['pred_recall']


def plot_confusion_matrix(true_labels: np.ndarray, predicted_labels: np.ndarray, labels: list, save_path: str) -> None:
    """
    Plots and saves the confusion matrix.

    Args:
        true_labels (np.ndarray): True labels.
        predicted_labels (np.ndarray): Predicted labels.
        labels (list): List of label names.
        save_path (str): Path to save the confusion matrix plot.
    """
    confusion = confusion_matrix(true_labels.squeeze(), predicted_labels)
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion, cmap='RdPu')
    plt.colorbar()
    plt.xlabel('Predicted Labels', fontsize=16)
    plt.ylabel('Actual Labels', fontsize=16)
    plt.title('Confusion Matrix', fontsize=16)

    total = np.sum(confusion) / len(labels)

    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            count = confusion[i, j]
            percent = (count / total) * 100
            text_color = 'white' if count > 50 else 'black'
            plt.text(j, i - 0.1, f'{count}', ha='center', va='center', color=text_color, fontsize=11)
            plt.text(j, i + 0.2, f'({percent:.1f}%)', ha='center', va='center', color=text_color, fontsize=9)

    tick_locations = np.arange(len(labels))
    plt.xticks(tick_locations, labels, rotation=45, ha='right', fontsize=9)
    plt.yticks(tick_locations, labels, rotation=45, ha='right', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_tsne(predicted: np.ndarray, true_labels: np.ndarray, num_classes: int, save_path: str) -> None:
    """
    Plots and saves the t-SNE visualization.

    Args:
        predicted (np.ndarray): Predicted outputs.
        true_labels (np.ndarray): True labels.
        num_classes (int): Number of classes.
        save_path (str): Path to save the t-SNE plot.
    """
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(predicted)

    plt.figure(figsize=(5, 5))
    plt.grid(True, ls='--', alpha=0.5)
    unique_labels = np.unique(true_labels)
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
    plt.savefig(save_path)
    plt.show()


def main():

    setup_logging()

    args = parse_arguments()
    logging.info(f'Parsed arguments: {args}')

    seed_func()
    logging.info('Seed set for reproducibility.')


    try:
        train_dataloader, test_dataloader, num_classes = load_dataset(args)
    except ValueError as e:
        logging.error(e)
        return


    net = MainNet()
    net = net.to(args.device)
    logging.info(f'Model {args.model_name} initialized and moved to {args.device}.')

    
    os.makedirs(args.path_weights, exist_ok=True)

    
    if args.train_mode:
        logging.info('Starting training phase...')
        full_loss, full_metric, best_model_name, best_acc, best_f1, best_recall = train_and_test_model(
            net=net,
            train_dataloader=train_dataloader,
            test_loader=test_dataloader,
            training_samples=args.training_samples_CWRU,
            num_epochs=args.num_epochs,
            lr=args.lr,
            loss1=ContrastiveLoss(),
            path_weight=args.path_weights,
            num_samples=args.training_samples_CWRU,
            num_classes=num_classes,
            model_name=args.model_name,
            device=torch.device(args.device)
        )
        logging.info('Training phase completed.')

    
    if args.cfs_matrix:
        logging.info('Starting validation phase...')
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

        # Load the best model
        best_model_path = os.path.join(args.path_weights, best_model_name)
        if not os.path.exists(best_model_path):
            logging.error(f'Best model file {best_model_path} does not exist.')
            return

        net.load_state_dict(torch.load(best_model_path))
        net = net.to(args.device)
        net.eval()
        logging.info(f'Best model {best_model_name} loaded for validation.')

        true_labels, predicted, vec_q, vec_s = predicted_fewshot(test_dataloader, net, args.device, num_classes)

        faults_labels = {v: k for k, v in faults_idx.items()}
        unique_labels = np.unique(true_labels)
        tick_labels = [faults_labels[label] for label in unique_labels]
        logging.info(f'Tick labels: {tick_labels}')

        predicted = predicted.squeeze()
        predicted_labels = np.argmax(predicted, axis=1)
        confusion = confusion_matrix(true_labels.squeeze(), predicted_labels)

        cfs_save_path = os.path.join(args.path_weights, f'cfs_{args.training_samples_CWRU}_{best_acc:.4f}.png')
        plot_confusion_matrix(true_labels, predicted_labels, tick_labels, cfs_save_path)

        tsne_save_path = os.path.join(args.path_weights, f'tsne_{args.training_samples_CWRU}_{best_acc:.4f}.png')
        plot_tsne(predicted, true_labels, num_classes, tsne_save_path)

        logging.info('Validation phase completed.')


if __name__ == '__main__':
    main()
