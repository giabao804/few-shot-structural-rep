import torch
import numpy as np
import torch.nn as nn
import argparse
from sklearn.model_selection import train_test_split
import function.function as function
from tqdm import tqdm
from function.function import ContrastiveLoss, seed_func, cal_metrics_5shot
from CWRU.CWRU_dataset import CWRU
from dataloader.dataloader import FewshotDataset
from torch.utils.data import DataLoader
from net.new_proposed import MainNet
import argparse
import torch.nn as nn
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from IPython.display import clear_output


parser = argparse.ArgumentParser(description='Bearing Faults Diagnosis')
parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
parser.add_argument('--h', type=int, default=16, help='Height of the input image')
parser.add_argument('--w', type=int, default=16, help='Width of the input image')
parser.add_argument('--c', type=int, default=64, help='Number of channels of the input image')
parser.add_argument('--dataset', choices=['HUST_bearing', 'CWRU', 'PDB'], help='Choose dataset for training')
parser.add_argument('--training_samples_CWRU', type=int, default=30, help='Number of training samples for CWRU')
parser.add_argument('--model_name', type=str, help='Model name')
parser.add_argument('--episode_num_train', type=int, default=130, help='Number of training episodes')
parser.add_argument('--episode_num_test', type=int, default=150, help='Number of testing episodes')
parser.add_argument('--way_num_CWRU', type=int, default=10, help='Number of classes for CWRU')
parser.add_argument('--noise_DB', type=str, default=None, help='Noise database')
parser.add_argument('--spectrum', action='store_true', help='Use spectrum')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda or cpu)')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--path_weights', type=str, default='checkpoints/', help='Path to weights')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--step_size', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--loss1', default=ContrastiveLoss())
parser.add_argument('--data_path', default="../few-shot-structural-rep/CWRU/", help="data path")
parser.add_argument('--cfs_matrix', action='store_false', help="Print confusion matrix")
parser.add_argument('--train_mode', action='store_false', help="Select train mode")
args = parser.parse_args()

print(args)
#---------------------------------------------------Load dataset-----------------------------------------------------------------------------------------:
if args.dataset == 'CWRU':
    window_size = 2048
    split = args.training_samples_CWRU//30
    data = CWRU(split, ['12DriveEndFault'], ['1772', '1750', '1730'], window_size)
    data.X_train = data.X_train.astype(np.float32)
    data.X_test = data.X_test.astype(np.float32)
    train_data_CWRU = torch.from_numpy(data.X_train)
    train_label_CWRU = torch.from_numpy(data.y_train)
    test_data_CWRU = torch.from_numpy(data.X_test)
    test_label_CWRU = torch.from_numpy(data.y_test)
    train_data_CWRU = train_data_CWRU.reshape([args.training_samples_CWRU,4096])
    test_data_CWRU = test_data_CWRU.reshape([750,4096])

    if args.noise_DB != None:
        snr_dB = args.noise_DB
        data.add_noise_to_test_data(snr_dB, 0.001)
        noisy_test_data = data.X_test_noisy.reshape([750,4096])

        if args.spectrum != None:
            train_data_CWRU = function.to_spectrum(train_data_CWRU)
            test_data_CWRU = function.to_spectrum(noisy_test_data)
        else:
            train_data_CWRU = train_data_CWRU.reshape(train_data_CWRU.shape[0], 1, 64, 64)
            test_data_CWRU = train_data_CWRU.reshape(test_data_CWRU.shape[0], 1, 64, 64)

    else:
        if args.spectrum != None:
            train_data_CWRU = function.to_spectrum(train_data_CWRU)
            test_data_CWRU = function.to_spectrum(test_data_CWRU)
        else:
            train_data_CWRU = train_data_CWRU.reshape(train_data_CWRU.shape[0], 1, 64, 64)
            test_data_CWRU = test_data_CWRU.reshape(test_data_CWRU.shape[0], 1, 64, 64)


    train_dataset_CWRU = FewshotDataset(train_data_CWRU, train_label_CWRU, episode_num=args.episode_num_train, way_num=args.way_num_CWRU, shot_num=5, query_num=1)
    train_dataloader_CWRU = DataLoader(train_dataset_CWRU, batch_size=args.batch_size, shuffle=True)
    test_dataset_CWRU = FewshotDataset(test_data_CWRU, test_label_CWRU, episode_num=args.episode_num_test, way_num=args.way_num_CWRU, shot_num=5, query_num=1)
    test_dataloader_CWRU = DataLoader(test_dataset_CWRU, batch_size=args.batch_size, shuffle=False)
    clear_output()
    # Testing phase
    print('Load_model_from_checkpoint.....')
    seed_func()
    net = MainNet().to(args.device)
    net = torch.load(args.best_weight)
    print('Loading successfully!')
    acc, f1, recall = cal_metrics_5shot(test_dataloader_CWRU, net, device = args.device)
    print('Accuracy on CWRU test data:', acc)

