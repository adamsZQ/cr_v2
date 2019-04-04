import sys

import torch
from sklearn.model_selection import train_test_split

import torch.utils.data as Data
from torch import nn, optim

from useless.lstm import load_model
from policy.policy_rl import Policy
from policy.pretrain.prepare_data import prepare_data
from tools.save_model import torch_save_model


def pre_train(bf_model_path_list, bf_prefix):
    # load five bf models
    bf_model_list = []
    for bf_model_path in bf_model_path_list:
        bf_model_list.append(load_model(bf_model_path))

    # TODO load data, data format is better like ( distribution of slot1, distribution of slot2...) and action
    state_list, action_list = prepare_data(bf_prefix)

    X_train, X_test, y_train, y_test = train_test_split(state_list, action_list, test_size=0.2, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    X_val = torch.from_numpy(X_val).float()
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()
    y_val = torch.from_numpy(y_val).long()

    batch_size = 32
    torch_dataset = Data.TensorDataset(X_train, y_train)
    loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=True)

    policy = Policy(len(state_list[0]), max(action_list) + 1)
    criterion = nn.CrossEntropyLoss(size_average=True)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    num_epochs = 100000
    best_loss = 1.6
    file_name = 'policy_pretrain'
    for epoch in range(num_epochs):
        for step, (batch_x, batch_y) in enumerate(loader):
            # forward
            out = policy(batch_x, train=True)  # forward
            loss = criterion(out, batch_y)  # loss

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 1 == 0:
            val_loss = val(X_val,y_val, policy, criterion)
            print('Epoch[{}/{}]'.format(epoch, num_epochs) + 'loss: {:.6f}'.format(
                loss.item()) + 'val_loss:{:.6f}'.format(val_loss))
            sys.stdout.flush()

            best_loss = torch_save_model(policy, bf_prefix, file_name, val_loss, best_loss)


def val(X_val, y_val, model, criterion):
    y_pred= model(X_val)
    val_loss = criterion(y_pred, y_val)  # 计算loss
    return val_loss.item()


def test(X_test, y_test, model, criterion):
    y_pred = model(X_test)
    loss = criterion(y_pred, y_test)  # 计算loss
    print(loss.item())
