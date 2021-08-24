import torch
import torch.nn as nn
import torch.optim as optim
import os
from utils.dataloader import ChestXRayDataset
from tqdm import tqdm
from utils.xception import XCeption
from utils.darkcovidnet import DarkCovidNet
import numpy as np


def create_loss_log_file(loss_log_file_name):

    f = open('log/' + loss_log_file_name + '_loss_log.txt', 'a')
    f.write('Log file start for the test: ' + loss_log_file_name + '_loss_log.txt\n')

    return f

def create_current_best_loss_file(best_loss_file_name):

    if (os.path.isfile('log/' +  best_loss_file_name + '_best_loss_log.txt')):
            f = open('log/' +  best_loss_file_name + '_best_loss_log.txt', "r+")
            lines = f.read().splitlines()
            if lines!=[]:
                best_loss = lines[-1]
                best_loss = float(best_loss)
            else:
                best_loss = 100.0

    else:
        f = open('log/' + best_loss_file_name + '_best_loss_log.txt', 'w')
        best_loss = 100.0

    return f, best_loss

def write_stats_after_epoch(loss_epoch, epoch, train_eval, file):
    print(train_eval + ', epoch: ' + str(epoch))
    print('Loss: ' + str(loss_epoch))
    print('')
    file.write(train_eval + ', lossEpoch' + str(epoch) + ', CELoss Mean: ' + str(loss_epoch) + '\n')

def load_model_and_optim(model_name, model_name_optimizer):

    torch_model = None
    torch_model_optim = None

    if (os.path.isfile(model_name) and os.path.isfile(model_name_optimizer)):
        torch_model = torch.load(model_name)
        torch_model_optim = torch.load(model_name_optimizer)

    return torch_model, torch_model_optim

def train_network(data_path, train_test, balancing_mode, architecture, batch_size_train, batch_size_eval, learning_rate):

    model_name = 'model__' + architecture + '__CE__' + str(target_resolution[0]) + '_' + str(target_resolution[1]) + '__b' + str(batch_size_train) + '__lr' + str(learning_rate) + "__bm" + balancing_mode + "__NOSOFTMAX" # resolution_batchSize_learningRate_balancingMode
    model_name_optimizer = model_name + '_optim'  # resolution_batchsize_learning_rate

    print("Used NN Architecture: " + architecture)
    print("Used batch size: " + str(batch_size_train))
    print("Used learning rate: " + str(learning_rate))

    loss_log_file = create_loss_log_file(model_name)
    best_loss_log_file, curr_best_eval_batch_loss = create_current_best_loss_file(model_name)
    torch_model, torch_model_optim = load_model_and_optim(model_name, model_name_optimizer)

    device = torch.device(0 if torch.cuda.is_available() else "cpu")

    if architecture == 'DarkCovid':
        model = DarkCovidNet(in_channels=3, num_labels=3, batch_size=batch_size_train, device=device)
    else:
        model = XCeption(device=device)
    model = model.double()
    model.to(device)

    if torch_model!=None:
        print("Continue training with stored model...")
        model.load_state_dict(torch_model, strict=False)  # enable if training continued

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if torch_model_optim!=None:
        print("Continue training with stored optimizer...")
        optimizer.load_state_dict(torch_model_optim)


    if train_test == 'train':

        train_dataset = ChestXRayDataset(os.path.join(data_path, train_test), target_resolution, 'train', balancing_mode)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, drop_last=True)

        eval_dataset = ChestXRayDataset(os.path.join(data_path, train_test), target_resolution, 'eval', balancing_mode)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size_eval, shuffle=True, drop_last=True)

        # label 0 -> normal | label 1 -> covid | label 2 -> pneumonia
        # train
        for epoch in range(100):
            model.train()
            epoch_train_loss = []
            for train_idx, (train_data, train_label) in enumerate(tqdm(train_dataloader, desc='Train')):
                train_label = train_label.to(device)
                train_data = train_data.to(device)
                train_data = train_data.double()

                if architecture == 'DarkCovid':
                    pred_label = model(train_data)
                else:
                    pred_label = model(train_data, batch_size_train)

                train_batch_loss = criterion(pred_label, train_label)

                optimizer.zero_grad()
                train_batch_loss.backward()
                optimizer.step()

                epoch_train_loss.append(train_batch_loss.item())
                loss_log_file.write('TrainLossEpoch' + str(epoch) + 'Step' + str(train_idx) + ': ' + str(train_batch_loss.item()) + '\n')


            print("Avg train batch loss, epoch: " + str(epoch) + ": " + str(np.mean(epoch_train_loss)))
            write_stats_after_epoch(np.mean(epoch_train_loss), epoch, 'Train', loss_log_file)

            if epoch % 1 == 0:
                model.eval()
                epoch_eval_loss = []
                for eval_idx, (eval_data, eval_label) in enumerate(tqdm(eval_dataloader, desc='Eval')):
                    eval_label = eval_label.to(device)
                    eval_data = eval_data.to(device)
                    eval_data = eval_data.double()

                    if architecture == 'DarkCovid':
                        eval_pred_label = model(eval_data)
                    else:
                        eval_pred_label = model(eval_data, batch_size_eval)

                    eval_batch_loss = criterion(eval_pred_label, eval_label)

                    epoch_eval_loss.append(eval_batch_loss.item())
                    loss_log_file.write('EvalLossEpoch' + str(epoch) + 'Step' + str(eval_idx) + ': ' + str(eval_batch_loss.item()) + '\n')

                if (np.mean(epoch_eval_loss) < curr_best_eval_batch_loss):
                    print("Current best eval batch loss: " + str(curr_best_eval_batch_loss))
                    print("New best eval batch loss: " + str(np.mean(epoch_eval_loss)))
                    print("Store model...")

                    curr_best_eval_batch_loss = np.mean(epoch_eval_loss)

                    torch.save(model.state_dict(), model_name)
                    torch.save(optimizer.state_dict(), model_name_optimizer)
                    best_loss_log_file.write(str(curr_best_eval_batch_loss) + '\n')
                    best_loss_log_file.flush()


                print("Avg eval batch loss, epoch: " + str(epoch) + ": " + str(np.mean(epoch_eval_loss)))

                write_stats_after_epoch(np.mean(epoch_eval_loss), epoch, 'Eval', loss_log_file)


        loss_log_file.close()


if __name__ == "__main__":

    data_path = ...  # insert absolute path to the data directory
    train_test = 'train'  # train or test

    architecture_arr = ['DarkCovid', 'xception']
    balancing_mode_arr = ["no", "balance_without_aug", "balance_with_aug"]
    learning_rate_arr = [0.001, 0.003, 0.005]

    for a in architecture_arr:
        for b in balancing_mode_arr:
            for lr in learning_rate_arr:

                if a == "DarkCovid":
                    target_resolution = (256, 256)
                    batch_size_train = 32
                    batch_size_eval = 32
                else:
                    target_resolution = (299, 299)  # modify here if other resolution needed
                    batch_size_train = 10
                    batch_size_eval = 10

                balancing_mode = b
                architecture = a
                learning_rate = lr
                train_network(data_path, train_test, balancing_mode, architecture, batch_size_train, batch_size_eval, learning_rate)

