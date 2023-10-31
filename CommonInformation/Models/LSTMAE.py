import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error

class Encoder(nn.Module):

    def __init__(self, input_size, embedding_size, use_bidirectional=False, num_layers=1):

        super().__init__()

        self.input_size = input_size  # number of expected modalities
        self.embedding_size = embedding_size  # bottleneck dimension
        self.use_bidirectional = use_bidirectional
        self.num_layers = num_layers
        if self.use_bidirectional:
            self.output_multiplication = 2
        else:
            self.output_multiplication = 1


        # first map from input_size dim to 2*embedding_size dim
        self.LSTM1 = nn.LSTM(
            input_size=self.input_size,
            hidden_size=2*self.embedding_size,
            num_layers=2,
            batch_first=True,
            bidirectional=self.use_bidirectional
        )

        # then map from 2*embedding_size dim dim to embedding_size dim
        self.LSTM2 = nn.LSTM(
            input_size=2 * self.output_multiplication * self.embedding_size,
            hidden_size=self.embedding_size,
            num_layers=2,
            batch_first=True,
            bidirectional=self.use_bidirectional
        )

    def forward(self, x):
        # input shape is (N, L, H_in) == (batch_num, seq_len, number_of_features)
        # inputs: input, (h_0, c_0) -> default to 0
        x, (hidden_state, cell_state) = self.LSTM1(x)
        x, (hidden_state, cell_state) = self.LSTM2(x)

        # hidden state is (num_layers, N, H_out) dimensional
        # hidden state is shaped (num_layers x self.output_multiplication, batch_size, embedding)
        # hidden state has hidden_states of multiple layers and multiple directions
        # take only the last forward and backward layers' hidden state
        last_lstm_layer_hidden_state = hidden_state.view(self.num_layers, self.output_multiplication, -1, self.embedding_size)
        last_lstm_layer_hidden_state = last_lstm_layer_hidden_state[-1]
        last_lstm_layer_hidden_state = last_lstm_layer_hidden_state.view(-1, self.output_multiplication * self.embedding_size)
        # concatenate forward and reverse hidden states
        return last_lstm_layer_hidden_state # (N, H_out) dimensional


class Decoder(nn.Module):
    def __init__(self, embedding_size, output_size, use_bidirectional=False, num_layers=1):
        super().__init__()

        self.embedding_size = embedding_size
        self.output_size = output_size
        self.use_bidirectional = use_bidirectional
        self.num_layers = num_layers
        if self.use_bidirectional:
            self.output_multiplication = 2
        else:
            self.output_multiplication = 1

        # mapping from embedding_size to embedding_size
        self.LSTM1 = nn.LSTM(
            input_size=self.output_multiplication * self.embedding_size,
            hidden_size=self.embedding_size,
            num_layers=2,
            batch_first=True,
            bidirectional=self.use_bidirectional
        )

        # mapping from embedding_size to 2*embedding_size
        self.LSTM2 = nn.LSTM(
            input_size=self.output_multiplication * self.embedding_size,
            hidden_size= 2*self.embedding_size,
            num_layers=2,
            batch_first=True,
            bidirectional=self.use_bidirectional
        )
        self.fc = nn.Linear(2*self.output_multiplication*self.embedding_size, output_size)

    def forward(self, x, seq_length):
        # x coming from encoder is (N, H_out) dimensional
        x = x.unsqueeze(1)  # now x is (N, 1, H_out) dimensional
        x = x.repeat(1, seq_length, 1)  # now it should be (N, seq_len, H_out) dimensional
        x, (hidden_state, cell_state) = self.LSTM1(x)
        x, (hidden_state, cell_state) = self.LSTM2(x)
        x = x.reshape((-1, seq_length, 2*self.embedding_size*self.output_multiplication))
        out = self.fc(x)
        return out

# now we wrap these together
class LSTM_AE(nn.Module):
    def __init__(self, input_size, embedding_dim, use_bidirectional=False, num_layers = 1,
                 train_on_gpu = True):
        super().__init__()

        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.use_bidirectional = use_bidirectional
        self.num_layers = num_layers
        self.output_size = input_size  # since we are reconstructing
        self.train_on_gpu = train_on_gpu
        self.encoder = Encoder(self.input_size, self.embedding_dim,
                               self.use_bidirectional, self.num_layers)
        self.decoder = Decoder(self.embedding_dim, self.output_size,
                               self.use_bidirectional, self.num_layers)

    def forward(self, x, seq_length):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded, seq_length)
        return encoded, decoded



def train(model, train_dataset, test_dataset, parameters):
    if (model.train_on_gpu):
        model.cuda()

    criterion = nn.MSELoss(reduction='mean')
    lr = parameters['lr']
    weight_decay = parameters['weight_decay']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    num_epochs = parameters['num_epochs']

    best_model_wts = None
    best_loss = 100000000.0
    best_epoch = -1

    train_losses = []
    test_losses = []
    for epoch in range(1, num_epochs + 1):
        # TRAIN
        avg_train_epoch_loss = train_one_epoch(model, train_dataset, optimizer, criterion, mode='train')
        avg_test_epoch_loss = train_one_epoch(model, test_dataset, optimizer, criterion, mode='test')
        train_losses.append(avg_train_epoch_loss)
        test_losses.append(avg_test_epoch_loss)

        if avg_test_epoch_loss < best_loss:
            print("model is changed!")
            best_loss = avg_test_epoch_loss
            best_model_wts = model
            best_epoch = epoch - 1
        print(f'Epoch {epoch}: Train loss: {avg_train_epoch_loss} Test: {avg_test_epoch_loss}')
    return best_model_wts.eval(), best_epoch, [train_losses, test_losses]

def train_one_epoch(model, dataset, optimizer, criterion, mode):
    losses = []
    if mode == 'train':
        model.train()
        for batch_idx, (inputs, targets) in enumerate(dataset):
            optimizer.zero_grad()
            inputs, targets = inputs.cuda(), targets.cuda()
            seq_length = inputs.size()[1]
            _, pred = model(inputs, seq_length)
            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().item())
    else:  # testing or validating
        model = model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataset):
                inputs, targets = inputs.cuda(), targets.cuda()
                seq_length = inputs.size()[1]
                _, pred = model(inputs, seq_length)
                loss = criterion(pred, targets)
                losses.append(loss.detach().item())
    return np.mean(np.array(losses))


def get_predictions(model, dataset, criterion):
    predictions, losses = [], []
    with torch.no_grad():
        model = model.eval()
        for batch_idx, (inputs, targets) in enumerate(dataset):
            inputs, targets = inputs.cuda(), targets.cuda()
            seq_length = inputs.size()[1]
            _, pred = model(inputs, seq_length)
            loss = criterion(pred, targets)

            predictions.append(pred)
            losses.append(loss.item())

    return predictions, losses

def encode_dataset(model, dataset):
    encodings = []
    with torch.no_grad():
        model = model.eval()
        for batch_idx, (inputs, _) in enumerate(dataset):
            inputs = inputs.cuda()
            encoded = model.encoder(inputs)
            encodings.append(encoded.cpu().numpy().flatten())
    return np.array(encodings)





