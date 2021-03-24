import pickle, random, string, sys, time, torch, os
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

random.seed(1)
np.random.seed(1)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
torch.manual_seed(1)
torch.backends.cudnn.benchmark = True

MAX_WORDS_SENTENCE = 25

# Model
class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.LSTM(4096, 256, batch_first = True)
        self.embedding = nn.Embedding(2428, 256)
        self.decoder = nn.LSTM(256*2, 256, batch_first = True)
        self.net = nn.Linear(256, 2428)

    def forward(self, X, y, train=True):
        output_list = []
        # Encoder
        encoder_output, hidden = self.encoder(X)
        idxs = torch.ones(X.shape[0], 1, dtype=torch.long).to(device)
        context = torch.zeros(X.shape[0], 1, 256).to(device)
        # Iterate over max length pre-defined for sentence
        for i in range(MAX_WORDS_SENTENCE):
            # Embed
            input_emb = torch.cat((self.embedding(idxs),context),2)
            # Decoder
            decoder_output, hidden = self.decoder(input_emb, hidden)
            # Attention
            attn_energy = torch.tanh(torch.bmm(encoder_output, hidden[0].transpose(0,1).transpose(1,2)))
            attn_weights = F.softmax(attn_energy, dim=1).squeeze(2).unsqueeze(1)
            context = torch.bmm(attn_weights, encoder_output)
            # Output index for word
            output = self.net(decoder_output)
            output_list.append(output)
            # Teacher Forcing
            teacher_force = random.random()
            if train and teacher_force <= 0.2:
                idxs = y[:,i].unsqueeze(1)
            else:
                _, idxs = torch.max(output, 2)
        return torch.cat(tuple(output_list), 1)


def train_model(model, X, data, optimizer, criterion):
    model.to(device)
    model.train()
    total_loss = 0
    for i, (X_pos, y) in enumerate(data):
        X_pos, y = X_pos.to(device), y.to(device)
        X_batch = X[X_pos.tolist(),:,:].to(device)
        optimizer.zero_grad()
        output = model(X_batch.float(), y, True)
        loss = criterion(output.view(-1, output.shape[-1]), y.view(-1))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 15)
        total_loss += loss
        loss.backward()
        optimizer.step()
    avg_loss = total_loss/len(data.dataset)
    return avg_loss.item()


if sys.argv[3] == "train":
    train_label = pd.read_json(sys.argv[2])

    # Dictionary for the count of each word
    words_count_dict = {}
    for i in range(train_label.shape[0]):
        labels = []
        for s in train_label['caption'][i]:
            # Remove characters from each sentence
            words_in_sentence = []
            for w in s.split(' '):
                words_in_sentence.append(w.strip(string.punctuation).lower())
            s = ' '.join(words_in_sentence)
            # Look at each word in sentence and add to dictionary
            for l in s.split(' '):
                words_count_dict[l] = words_count_dict.get(l, 0) + 1
            if len(s.split(' ')) < MAX_WORDS_SENTENCE:
                labels.append(s)
        train_label['caption'][i] = labels

    # Create list of words that occur more than 3 times, 2424 words remain (+4 tags) from a total of 6057
    words = ['<pad>', '<bos>', '<eos>', '<unk>']
    words += [w for w in words_count_dict if words_count_dict[w] > 3]

    # Save words so it can be reopened during testing
    with open('resources/words.obj', 'wb') as out: 
        pickle.dump(words, out)

    # List of training features
    train_feature_directory = sys.argv[1]
    train_feature_list = []
    for i in train_label['id']:
        train_feature_list.append(np.load(train_feature_directory + i + ".npy"))

    # Prepare data
    X_positions = []
    y = []
    for id_pos in range(1450):
        for lbl in train_label['caption'][id_pos]:
            X_positions.append(id_pos)
            y.append([3 if i not in words else words.index(i) for i in lbl.split()] + [2])
    y = torch.LongTensor(np.array([i+[2]*(MAX_WORDS_SENTENCE-len(i)) for i in y]))

    train_loader = Data.DataLoader(Data.TensorDataset(torch.tensor(X_positions), y), 64, shuffle=True)

    # Train Features
    X = torch.tensor(train_feature_list)

    # Call Model
    model = Seq2Seq().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    loss_list = []
    for i in range(1,81):
        start = time.time()
        loss = train_model(model, X, train_loader, optimizer, criterion)
        print('Epoch:{}  \tLoss:{:.8f}\t\tTime:{:.4f}s'.format(i, loss, time.time()-start))
        loss_list.append(loss)
    torch.save(model.cpu().state_dict(), 'resources/model.pt')


elif sys.argv[3] == 'test':
    output_name = sys.argv[2]
    test_feature_directory = sys.argv[1]
    files = os.listdir(test_feature_directory)

    # Get word list
    with open('resources/words.obj', 'rb') as f: 
        words = pickle.load(f)
    
    # Test features
    feature_list = []
    for i in files:
        feature_list.append(np.load(test_feature_directory + i))
    
    # Load model
    model = Seq2Seq().to(device)
    model.load_state_dict(torch.load('resources/model.pt'))
    video_prediction_list = []
    model.eval()
    
    # Get predictions
    for i in range(len(feature_list)):
        feat = torch.tensor(feature_list[i]).unsqueeze(0).to(device)
        output = model(feat.float(), None, False).squeeze(0)
        _, idx_list = torch.max(output,1)
        video_prediction_list.append([files[i][:-4], ' '.join([words[i] for i in idx_list.tolist() if i != 0 and i != 2])])

    # Write results to file
    with open(output_name,'w') as out:
        for i in video_prediction_list:
            out.write('{},{}\n'.format(i[0], i[1]))
