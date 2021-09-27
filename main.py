import torch
import torch.nn as nn
import torch.optim as optim
import os
from utils.dataloader import ChestXRayDataset
from tqdm import tqdm
from utils.xception import XCeption
from utils.darkcovidnet import DarkCovidNet
from utils.densenet import DenseNet, DENSENET169, DENSENET201, DENSENET264, DENSENET121
import numpy as np
import utils.eval_metrics as eval_metrics


def create_loss_log_file(loss_log_file_name):
    """
    Creates loss log file which stores loss after every batch in the training.
    Parameters
    ----------
    loss_log_file_name:     str
                            A name of newly created loss log file

    Returns
    -------
    f:                      file
                            Newly created file
    """

    f = open('log/' + loss_log_file_name + '_loss_log.txt', 'a')
    f.write('Log file start for the test: ' + loss_log_file_name + '_loss_log.txt\n')

    return f


def create_current_best_loss_file(best_loss_file_name):
    """
    Creates loss log file which stores epoch loss only if better than current best loss.
    Parameters
    ----------
    best_loss_file_name:    str
                            A name of newly created best loss log file

    Returns
    -------
    f:                      file
                            Newly created file
    best_loss:              float
                            Current best loss
    """

    if (os.path.isfile('log/' + best_loss_file_name + '_best_loss_log.txt')):
        f = open('log/' + best_loss_file_name + '_best_loss_log.txt', "r+")
        lines = f.read().splitlines()
        if lines != []:
            best_loss = lines[-1]
            best_loss = float(best_loss)
        else:
            best_loss = 100.0
    else:
        f = open('log/' + best_loss_file_name + '_best_loss_log.txt', 'w')
        best_loss = 100.0

    return f, best_loss


def write_stats_after_epoch(loss_epoch, epoch, train_eval, file):
    """
    Writes loss in the file after every epoch
    Parameters
    ----------
    loss_epoch:             float
                            Loss in the current epoch
    epoch:                  int
                            Current epoch
    train_eval:             str
                            Train or Eval mode
    file:                   file
                            File to be written
    """

    print(train_eval + ', epoch: ' + str(epoch))
    print('Loss: ' + str(loss_epoch))
    print('')
    file.write(train_eval + ', lossEpoch' + str(epoch) + ', CELoss Mean: ' + str(loss_epoch) + '\n')


def load_model_and_optim(model_name, model_name_optimizer):
    """
    Loads model and optimizer if exist (continues the training). Else None returned, new training initialized.
    Parameters
    ----------
    model_name:             str
                            A name of the model to be loaded
    model_name_optimizer:   str
                            A name of the optimizer to be loaded

    Returns
    -------
    torch_model:            object
                            Loaded model or None it doesn't exist
    torch_model_optim:      object
                            Loaded model optimizer or None it doesn't exist
    """

    torch_model = None
    torch_model_optim = None

    if (os.path.isfile(model_name) and os.path.isfile(model_name_optimizer)):
        torch_model = torch.load(model_name)
        torch_model_optim = torch.load(model_name_optimizer)

    return torch_model, torch_model_optim


def init_model_criterion_optimizer(model_name, model_name_optimizer, device):
    """
    Initializes model, criterion and optimizer for the network.
    Parameters
    ----------
    model_name:             str
                            A name of the model to be loaded
    model_name_optimizer:   str
                            A name of the optimizer to be loaded
    device:                 device
                            Device used for the training (CUDA or cpu), depending on the hardware

    Returns
    -------
    model:                  object
                            Initialized model
    criterion:              object
                            Initialized criterion
    optimizer:              object
                            Initialized optimizer
    """

    torch_model, torch_model_optim = load_model_and_optim(model_name, model_name_optimizer)

    if architecture == 'DarkCovid':
        model = DarkCovidNet(in_channels=3, num_labels=3, device=device)
    elif architecture == 'DenseNet':
        model = DenseNet(device=device, architecture=DENSENET121)
    else:
        model = XCeption(device=device)

    model = model.double()
    model.to(device)

    if torch_model != None:
        print("Continue training with stored model...")
        model.load_state_dict(torch_model, strict=False)

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if torch_model_optim != None:
        print("Continue training or test with stored optimizer...")
        optimizer.load_state_dict(torch_model_optim)

    return model, criterion, optimizer


def train_eval_test_batch(label, data, device, model, criterion, train=True):
    """
    Performs tasks needed in every batch of the training/eval/test of the network. Returns batch_loss if train, else predicted label for the test.
    Parameters
    ----------
    label:                  int
                            Class label used for the evaluation of the network prediction (0, 1 or 2)
    data:                   tensor
                            Torch tensor including batch of the images
    device:                 device
                            Device used for the training (CUDA or cpu), depending on the hardware
    model:                  object
                            Model used for the prediction (for the training)
    criterion:              object
                            Criterion used for the calculating the loss (Cross Entropy Loss)
    train:                  bool
                            Training or test mode (True if training)

    Returns
    -------
    batch_loss:             object
                            Batch loss used for the training and eval mode
    pred_label:             tensor
                            Torch tensor with predicted values for the test
    """

    label = label.to(device)
    data = data.to(device)
    data = data.double()

    if architecture == 'DarkCovid':
        pred_label = model(data)
    elif architecture == 'DenseNet':
        pred_label = model(data)
    else:
        pred_label = model(data, batch_size_train)

    batch_loss = criterion(pred_label, label)

    if train:
        return batch_loss
    else:
        return pred_label

def train_test_network(data_path, train_test, balancing_mode, architecture, batch_size_train, batch_size_eval,
                       learning_rate):
    """
    Trains and tests the network over specific number of epochs
    Parameters
    ----------
    data_path:              str
                            Absolute path to the directory which contains data (X-Ray images)
    train_test:             str
                            train or test mode of the network
    balancing_mode:         str
                            Balancing mode, can be: no, balance_without_aug or balance_with_aug
    architecture:           str
                            Architecture used for the training, can be: DenseNet, DarkCovid or Xception
    batch_size_train:       int
                            Batch size used for the training mode
    batch_size_eval:        int
                            Batch size used for the eval mode
    learning_rate:          float
                            Learning rate used for the training
    """

    model_name = 'model__' + architecture + '__CE__' + str(target_resolution[0]) + '_' + str(
        target_resolution[1]) + '__b' + str(batch_size_train) + '__lr' + str(
        learning_rate) + "__bm" + balancing_mode + "__NOSOFTMAX"  # resolution_batchSize_learningRate_balancingMode

    model_name_optimizer = model_name + '_optim'

    if train_test == 'train':
        print("Used NN Architecture: " + architecture)
        print("Used batch size: " + str(batch_size_train))
        print("Used learning rate: " + str(learning_rate))

        loss_log_file = create_loss_log_file(model_name)
        best_loss_log_file, curr_best_eval_batch_loss = create_current_best_loss_file(model_name)
    else:
        print("Test mode activated...")

    device = torch.device(0 if torch.cuda.is_available() else "cpu")

    model, criterion, optimizer = init_model_criterion_optimizer(model_name, model_name_optimizer, device)

    if train_test == 'train':

        train_dataset = ChestXRayDataset(os.path.join(data_path, train_test), target_resolution, 'train',
                                         balancing_mode)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True,
                                                       drop_last=True)

        eval_dataset = ChestXRayDataset(os.path.join(data_path, train_test), target_resolution, 'eval', balancing_mode)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size_eval, shuffle=True,
                                                      drop_last=True)

        # label 0 -> normal | label 1 -> covid | label 2 -> pneumonia
        # train
        for epoch in range(100):
            model.train()
            epoch_train_loss = []
            for train_idx, (train_data, train_label) in enumerate(tqdm(train_dataloader, desc='Train')):

                train_batch_loss = train_eval_test_batch(train_label, train_data, device, model, criterion)

                epoch_train_loss.append(train_batch_loss.item())
                loss_log_file.write('TrainLossEpoch' + str(epoch) + 'Step' + str(train_idx) + ': ' + str(
                    train_batch_loss.item()) + '\n')

                optimizer.zero_grad()
                train_batch_loss.backward()
                optimizer.step()

            print("Avg train batch loss, epoch: " + str(epoch) + ": " + str(np.mean(epoch_train_loss)))
            write_stats_after_epoch(np.mean(epoch_train_loss), epoch, 'Train', loss_log_file)

            # eval network only every n-th epoch
            if epoch % 1 == 0:
                model.eval()
                epoch_eval_loss = []
                for eval_idx, (eval_data, eval_label) in enumerate(tqdm(eval_dataloader, desc='Eval')):

                    eval_batch_loss = train_eval_test_batch(eval_label, eval_data, device, model, criterion)

                    epoch_eval_loss.append(eval_batch_loss.item())
                    loss_log_file.write('EvalLossEpoch' + str(epoch) + 'Step' + str(eval_idx) + ': ' + str(
                        eval_batch_loss.item()) + '\n')

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

    else:
        # test
        model.eval()
        test_dataset = ChestXRayDataset(os.path.join(data_path, train_test), target_resolution, 'test', balancing_mode)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_eval, shuffle=True,
                                                      drop_last=True)
        all_data_test_label = np.array([], dtype=np.int64)
        all_data_test_pred_label = np.array([], dtype=np.int64)
        for test_idx, (test_data, test_label) in enumerate(tqdm(test_dataloader, desc='Test')):

            test_pred_label = train_eval_test_batch(test_label, test_data, device, model, criterion, train=False)

            test_pred_label_arg_max = np.array(torch.argmax(torch.softmax(test_pred_label, 1), 1).cpu())  # as npy
            test_label = np.array(test_label.cpu())  # as npy


            all_data_test_label = np.hstack([all_data_test_label, test_label])
            all_data_test_pred_label = np.hstack([all_data_test_pred_label, test_pred_label_arg_max])

        eval_metrics.get_evaluation_metrics(all_data_test_pred_label, all_data_test_label, model_name + '_evaluation_metrics')
        eval_metrics.get_confusion_matrix(all_data_test_pred_label, all_data_test_label, model_name + '_confusion_matrix.jpg')


if __name__ == "__main__":

    data_path = ...  # insert absolute path to the data directory
    train_test = 'train'  # train or test

    architecture_arr = ['DenseNet', 'DarkCovid', 'xception']
    balancing_mode_arr = ["no", "balance_without_aug", "balance_with_aug"]
    learning_rate_arr = [0.001, 0.003, 0.005]


    for a in architecture_arr:
        for b in balancing_mode_arr:
            for lr in learning_rate_arr:

                if a == "DarkCovid":
                    target_resolution = (256, 256)
                    batch_size_train = 32
                    batch_size_eval = 32
                elif a == "DenseNet":
                    target_resolution = (224, 224)
                    batch_size_train = 6
                    batch_size_eval = 6

                else:
                    target_resolution = (299, 299)  # modify here if other resolution needed
                    batch_size_train = 10
                    batch_size_eval = 10

                balancing_mode = b
                architecture = a
                learning_rate = lr
                train_test_network(data_path, train_test, balancing_mode, architecture, batch_size_train, batch_size_eval,
                                   learning_rate)
