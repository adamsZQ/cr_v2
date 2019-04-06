import os
import sys

import h5py

import numpy as np
import torch
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch import nn, optim
from torch.autograd import Variable
import torch.utils.data as Data

from policy.policy_rl import Policy

data_file = '/home/next/cr_repo/pre_train/pretrain_data.h5'
f = h5py.File(data_file, 'r')
states = f['states'][:]
actions = f['actions'][:]
actions = np.squeeze(actions, axis=1)
# actions_tc_list = []
# for action in actions:
#     actions_tc = [0] * (np.max(actions)+1)
#     actions_tc[action] = 1
#     actions_tc_list.append(actions_tc)

actions_tc_list = np.array(actions)

input_size = states.shape[1]
output_size = np.max(actions_tc_list) + 1

# inputs = torch.from_numpy(states).float()
# target = torch.from_numpy(actions_tc_list).float()

X_train, X_test, y_train, y_test = train_test_split(states, actions_tc_list, test_size=0.2, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
X_val = torch.from_numpy(X_val).float()
y_train = torch.from_numpy(y_train).long()
y_test = torch.from_numpy(y_test).long()
y_val = torch.from_numpy(y_val).long()


# 添加批训练
batch_size = 32
torch_dataset = Data.TensorDataset(X_train, y_train)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=True)

policy = Policy(input_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(policy.parameters(), lr=1e-4)


def train(model_prefix, file_name):
    num_epochs = 100000
    best_loss = 0.3
    for epoch in range(num_epochs):
        for step, (batch_x, batch_y) in enumerate(loader):
            # forward
            out = policy(batch_x, train=True)  # 前向传播 TODO default is including Softmax Wrong!!!!!!!
            loss = criterion(out, batch_y)  # 计算loss

            # backward
            optimizer.zero_grad()  # 梯度归零
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

        if epoch % 1 == 0:
            accuracy, precision, recall, f1 = val(X_val, y_val)
            print('Epoch[{}/{}]'.format(epoch, num_epochs) + 'loss: {:.6f}'.format(
                loss.item()) )
            print('accuracy: {:.6f}'.format(accuracy) +
                  'precision: {:.6f}'.format(precision) +
                  'recall: {:.6f}'.format(recall) +
                  'f1: {:.6f}'.format(f1)
                  )
            sys.stdout.flush()

            best_loss = save_model(policy, model_prefix, file_name, 1 - f1, best_loss)


def val(X_val, y_val):
    predict_list = []
    target_list = []
    for state, tags in zip(X_val,y_val):
        # sentence = torch.tensor(state).long()
        state = state.unsqueeze(0)
        predict = policy(state, train=False)
        # print(predict)
        predict_list.append([np.argmax(predict.tolist())])
        target_list.append([tags.tolist()])

    binarizer = MultiLabelBinarizer()
    binarizer.fit_transform([[x for x in range(policy.output_size)]])
    target_list = binarizer.transform(target_list)
    predict_list = binarizer.transform(predict_list)

    accuracy = accuracy_score(target_list, predict_list)
    f1 = f1_score(target_list, predict_list, average="weighted")
    precision = precision_score(target_list, predict_list, average="macro")
    recall = recall_score(target_list, predict_list, average="macro")

    return accuracy, precision, recall, f1


def test(X_test, y_test):
    y_pred = policy(X_test)
    loss = criterion(y_pred, y_test)  # 计算loss
    print(loss.item())


def save_model(model, file_prefix=None, file_name=None, val_loss='None', best_loss='None', enforcement = False):
    # Save model
    try:
        if enforcement or val_loss == 'None' or best_loss == 'None':
            file_path = '{}{}_{}.pkl'.format(file_prefix, file_name, 'enforcement')
            torch.save(model, file_path)
            print('enforcement save:', file_path)

        elif val_loss != 'None' and best_loss != 'None' and ~enforcement:
            is_best = val_loss < best_loss
            best_loss = min(best_loss, val_loss)
            if is_best:
                file_path = '{}{}_{:.4f}.pkl'.format(file_prefix, file_name, best_loss)
                torch.save(model, file_path)
            return best_loss
    except Exception as e:
        # if error, save model in default path
        print(e)
        file_path = '{}{}.pkl'.format(os.getcwd(), '/default')
        print('default save:', file_path)
        torch.save(model, file_path)


if __name__ == '__main__':
    FILE_PREFIX = '/home/next/cr_repo/pre_train/'
    file_name = 'policy_pretrain'
    train(FILE_PREFIX, file_name)

    # policy = torch.load(FILE_PREFIX+file_name)
    # test(X_test, y_test)
    '''
    pretrain result:val_loss:0.7437, test_loss:0.7498
    '''



