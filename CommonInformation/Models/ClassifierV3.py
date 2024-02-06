import torch.nn as nn
import torch
import numpy as np

# only one global model for classification but modalities are processed differently


class ClassifierNet(nn.Module):
    def __init__(self, common_rep_dim, hidden_1, output_dim, train_on_gpu=True):
        super(ClassifierNet, self).__init__()
        self.common_rep_dim = common_rep_dim
        self.output_dim = output_dim
        self.fc = nn.Sequential(nn.Linear(self.common_rep_dim, hidden_1),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(hidden_1, self.output_dim))
        self.train_on_gpu = train_on_gpu

    def forward(self, x):
        return self.fc(x)



def train_one_epoch(loader, optimizers, models, criterion, modalities, mode):
    batch_losses = {}
    all_ground_truth = {}
    all_predictions = {}
    all_prediction_confidences = {}
    for modality in modalities:
        batch_losses[modality] = []
        all_ground_truth[modality] = []
        all_predictions[modality] = []
        all_prediction_confidences[modality] = []

    if mode == 'train':
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs_list = np.split(inputs, len(modalities), axis=-1)
            # zero the parameter gradients
            for modality_idx in range(len(modalities)):
                modality = modalities[modality_idx]
                input_modality = torch.squeeze(inputs_list[modality_idx])
                optimizers[modality].zero_grad()
                # get the inputs; data is a list of [inputs, labels]
                modality_loss, all_ground_truth[modality], all_predictions[modality], all_prediction_confidences[modality] = forward_pass(input_modality, targets,
                                                                       models['base'], models['common'], models['classification'],
                                                                       criterion, all_ground_truth[modality], all_predictions[modality], all_prediction_confidences[modality])
                modality_loss.backward()
                optimizers[modality].step()
                batch_losses[modality].append(modality_loss.detach().item())
    else:  # val or test
        for batch_idx, (inputs, targets) in enumerate(loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs_list = np.split(inputs, len(modalities), axis=-1)
            for modality_idx in range(len(modalities)):
                modality = modalities[modality_idx]
                input_modality = torch.squeeze(inputs_list[modality_idx])
                # get the inputs; data is a list of [inputs, labels]
                modality_loss, all_ground_truth[modality], all_predictions[modality], all_prediction_confidences[modality] = forward_pass(input_modality, targets,
                                                                       models['base'], models['common'], models['classification'],
                                                                       criterion, all_ground_truth[modality], all_predictions[modality], all_prediction_confidences[modality])
                batch_losses[modality].append(modality_loss.detach().item())

    return batch_losses, all_ground_truth, all_predictions, all_prediction_confidences

def forward_pass(inputs, targets, base_net, common_net, classification_net, criterion,
                 all_ground_truth, all_predictions, all_prediction_confidences):

    inputs, targets = inputs.cuda(), targets.cuda()
    outputs = base_net(inputs)
    outputs = common_net(outputs)

    outputs = classification_net(outputs)
    loss = criterion(outputs, torch.max(targets, 1)[1])

    top_p, top_class = outputs.topk(1, dim=1)
    all_ground_truth = all_ground_truth + torch.max(targets, 1)[1].cuda().view(-1).tolist()
    all_predictions = all_predictions + top_class.view(-1).tolist()
    all_prediction_confidences = all_prediction_confidences + top_p.view(-1).tolist()


    return loss, all_ground_truth, all_predictions, all_prediction_confidences