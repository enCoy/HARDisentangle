import torch.nn as nn
import torch



class ClassifierNet(nn.Module):
    def __init__(self, common_rep_dim, hidden_1, output_dim, train_on_gpu=True):
        super(ClassifierNet, self).__init__()
        self.common_rep_dim = common_rep_dim
        self.output_dim = output_dim
        self.fc = nn.Sequential(nn.Linear(self.common_rep_dim, hidden_1),
                                nn.ReLU(),
                                nn.Linear(hidden_1, self.output_dim))
        self.train_on_gpu = train_on_gpu

    def forward(self, x):
        return self.fc(x)



def train_one_epoch(loader, optimizer, base_net, common_net, classification_net, criterion, mode):
    batch_losses = []
    if mode == 'train':
        for batch_idx, (inputs, targets) in enumerate(loader):
            # zero the parameter gradients
            optimizer.zero_grad()
            # get the inputs; data is a list of [inputs, labels]
            loss = forward_pass(inputs, targets, base_net, common_net, classification_net, criterion)
            batch_losses.append(loss.detach().item())
            loss.backward()
            optimizer.step()
    else:  # val or test
        for batch_idx, (inputs, targets) in enumerate(loader):
            # get the inputs; data is a list of [inputs, labels]
            loss = forward_pass(inputs, targets, base_net, common_net, classification_net, criterion)
            batch_losses.append(loss.detach().item())
    return batch_losses

def forward_pass(inputs, targets, base_net, common_net, classification_net, criterion):
    inputs, targets = inputs.cuda(), targets.cuda()
    outputs = base_net(inputs)
    outputs = common_net(outputs)
    outputs = classification_net(outputs)
    loss = criterion(outputs, torch.max(targets, 1)[1])
    return loss