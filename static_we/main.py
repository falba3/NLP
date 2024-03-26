import torch
import torch.nn as nn
from torch.optim import Adam
from torchtext.vocab import GloVe
from nltk.tokenize import word_tokenize
from sklearn.metrics import f1_score
from datasets import load_dataset
import torch.nn.functional as F
import pandas as pd

torch.manual_seed(0) # Setting a seed to replicate results (random initialization of weights)
dataset = load_dataset("rotten_tomatoes")

glove = GloVe(name='6B', dim=50, cache="/tmp/glove/") # using the GloVe static embeddings
word_index = {word: idx + 1 for idx, word in enumerate(glove.itos)}
embedding_index = {word: glove.vectors[glove.stoi[word]] for word in glove.stoi}
embedding_matrix = glove.vectors # creating an embedding matrix to be passed onto the network

# Splitting Train and Test sets
X_train = dataset['train']['text']
y_train = dataset['train']['label']
X_test = dataset['test']['text']
y_test = dataset['test']['label']


def tokenize_and_index(sentences, max_length):
    """
    :param sentences: strings of the reviews
    :param max_length: max length of sequences to be allowed (I set to 100 after seeing
    that there is no sentence with more than 100 words)
    :return: indices for the embeddings of tokens found in GloVe
    """
    indexed_sentences = []
    for sentence in sentences:
        tokens = word_tokenize(sentence.lower())
        indices = [glove.stoi.get(token, 0) for token in tokens]  # Uses index 0 for unknown tokens
        indices = indices[:max_length]  # Truncate or pad to max_length
        padded_indices = indices + [0] * (max_length - len(indices))  # To pad with zeros
        indexed_sentences.append(padded_indices)
    return indexed_sentences

def train_model(model, X_train, y_train, epochs, lr=0.001):
    """
    :param model: the neural network model to be trained
    :param X_train: a tensor of the set of embedding indices for reviews in the train set
    :param y_train: a tensor of the labels in the train set
    :param epochs: how many epochs the model will be trained over
    :param lr: hyperparameter learning rate of the optimizer
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train.float())
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

def evaluate_model(model, X_test, y_test):
    """
    :param model: the neural network model to be evaluated
    :param X_test: a tensor of the set of embedding indices for reviews in the test set
    :param y_test: a tensor of the labels in the est set
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predictions = torch.round(torch.sigmoid(outputs))
        accuracy = (predictions.squeeze() == y_test).sum().item() / len(y_test)
        f1 = f1_score(y_test.cpu().numpy(), predictions.squeeze().cpu().numpy())
        print(f"Accuracy: {accuracy}, F1 score: {f1}")

def create_preds(model, X_test):
    """
    :param model: the neural network model that will create predictions
    :param X_test: a tensor of the set of embedding indices for reviews in the test set to predict from
    :return: pandas DataFrame of the predictions with index column
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predictions = torch.round(torch.sigmoid(outputs))
        predictions = predictions.squeeze()

        pred_df = pd.DataFrame({'pred': predictions})
        index_df = pd.DataFrame(list(range(0, len(predictions))), columns=['index'])
        pred_df = pd.concat([index_df, pred_df], axis=1)
        return pred_df

# The class definition for the best model that I trained on with the corresponding parameters
class StaticEmbeddingCNN(nn.Module):
    def __init__(self, embedding_matrix, num_classes, kernel_sizes=[3, 4, 5], num_filters=100, dropout_prob=0.5):
        super(StaticEmbeddingCNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_matrix.shape[1], out_channels=num_filters, kernel_size=ks)
            for ks in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(0, 2, 1)

        conv_outputs = [F.relu(conv(embedded)) for conv in self.convs]

        pooled_outputs = [F.max_pool1d(conv_output, conv_output.size(2)).squeeze(2) for conv_output in conv_outputs]  # Pooling

        concat = torch.cat(pooled_outputs, dim=1)

        output = self.dropout(concat)
        output = self.fc(output)
        return output


def main():
    """
    1. Tokenize and index the train and test sets.
        Train set will be converted to indices of the corresponding embeddings.
    2. Conversion to PyTorch tensors.
    3. Instantiation of model and training over 100 epochs.
    4. Evaluation of model.
    5. Creation of predictions.
    6. Saving of predictions to 'results.csv'

    """
    max_length = 100
    X_train_indices = tokenize_and_index(X_train, max_length)
    X_test_indices = tokenize_and_index(X_test, max_length)

    X_train_tensor = torch.tensor(X_train_indices, dtype=torch.long)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_indices, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    num_classes = 1
    model = StaticEmbeddingCNN(embedding_matrix, num_classes)
    train_model(model, X_train_tensor, y_train_tensor, epochs=100)
    evaluate_model(model, X_test_tensor, y_test_tensor)

    out = create_preds(model, X_test_tensor)
    out.to_csv("results.csv")

if __name__ == "__main__":
    try:
        # If there is any problem with main during training, I will load my saved best model
        main()
    except:
        # The abovementioned CNN trained on 100 epochs and generate the results
        print("Loading saved model...")
        loaded_model = torch.load("saved_models/CNN.pth")
        max_length = 100
        X_test_indices = tokenize_and_index(X_test, max_length)
        X_test_tensor = torch.tensor(X_test_indices, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        evaluate_model(loaded_model, X_test_tensor, y_test_tensor)
        out = create_preds(loaded_model, X_test_tensor)
        out.to_csv("results.csv")



