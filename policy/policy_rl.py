import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=20):
        super(Policy, self).__init__()
        # state -> hidden
        self.affine1 = nn.Linear(input_size, hidden_size)
        # hidden -> action
        self.affine2 = nn.Linear(hidden_size, output_size)

        self.saved_log_probs = []
        self.rewards = []

        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)
        self.eps = np.finfo(np.float32).eps.item()

        self.gamma = 0.99

    def forward(self, x, train):
        if train:
            # when training remove softmax
            model = torch.nn.Sequential(
                self.affine1,
                # TODO 需要检验失活函数是否需要
                # nn.Dropout(p=0.5),
                nn.ReLU(),
                self.affine2,
                nn.ReLU(),
            )
        else:
            # when predicting, put softmax back
            model = torch.nn.Sequential(
                self.affine1,
                # TODO 需要检验失活函数是否需要
                # nn.Dropout(p=0.5),
                nn.ReLU(),
                self.affine2,
                nn.ReLU(),
                nn.Softmax()
            )
        return model(x)

    def select_action(self,state, device):
        # when training, activate this function
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self(state, train=True)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def select_best_action(self,state, device):
        # when predicting, activate this function
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self(state, train=False)
        action = torch.argmax(probs, dim=1).tolist()
        return action[0]

    def update_policy(self, optimizer):
        R = 0
        policy_loss = []
        rewards = []
        for r in self.rewards[::-1]:
            # calculate R discounted
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        if len(rewards) == 1:
            # TODO if rewards length is one , do nothing
            pass
        else:
            # normalize rewards
            rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)
        for log_prob, reward in zip(self.saved_log_probs, rewards):
            # loss = sum(-log_prob * reward)
            policy_loss.append(-log_prob * reward)

        # loss = sum(-log_prob * reward)
        policy_loss = torch.cat(policy_loss).sum()
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        # clean rewards and log probability
        del self.rewards[:]
        del self.saved_log_probs[:]

    def add_reward(self, reward):
        # add reward to reward sequence
        self.rewards.append(reward)


if __name__ == '__main__':
    state1 = np.array([-1,-1,-1,-1, -1])
    state2 = np.array([57,-1,-1,-1, -1])
    state3 = np.array([-1,-1,-1,-1, -1])
    # state = torch.from_numpy(state).float().unsqueeze(0)

    FILE_PREFIX = '/path/mv/model/'
    file_name = 'policy_pretrain_0.7437.pkl'
    model = torch.load(FILE_PREFIX + file_name)

    a = torch.tensor(state1).float()
    b = a.std()
    mean = a.mean()

    for i in range(300):
        action1 = model.select_action(np.array(state1))
        print('action1', action1)
        model.rewards.append(-1)
        action2 = model.select_action(np.array(state2))
        print('action2', action2)
        model.rewards.append(-30)
        optimizer = optim.RMSprop(model.parameters(), lr=1e-4)

        model.update_policy(optimizer)





