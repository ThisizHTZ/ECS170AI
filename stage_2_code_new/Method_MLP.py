'''
Concrete MLP method for stage 2 multiclass classification
'''

from Base_Classes import method
import torch
from torch import nn
import numpy as np
import copy
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Method_MLP(method, nn.Module):
    data = None

    max_epoch = 30
    learning_rate = 1e-3
    batch_size = 128
    device = None

    def __init__(
        self,
        mName,
        mDescription,
        hidden_dims=None,
        dropout_rate=0.25,
        use_batchnorm=True,
        weight_decay=1e-4,
        patience=5,
        seed=42
    ):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.seed = seed
        set_seed(self.seed)

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm
        self.weight_decay = weight_decay
        self.patience = patience

        self.network = self._build_network()

        if self.device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')

        self.to(self.device)

        self.train_loss_history = []
        self.train_acc_history = []
        self.test_loss_history = []
        self.test_acc_history = []
        self.best_epoch = None

    def _build_network(self):
        layers = []
        input_dim = 784

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))

            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())

            if self.dropout_rate > 0:
                layers.append(nn.Dropout(p=self.dropout_rate))

            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 10))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def _get_batches(self, X, y, batch_size):
        n = len(X)
        indices = np.arange(n)
        np.random.shuffle(indices)

        for start_idx in range(0, n, batch_size):
            end_idx = min(start_idx + batch_size, n)
            batch_idx = indices[start_idx:end_idx]
            yield X[batch_idx], y[batch_idx]

    def evaluate_dataset(self, X, y, loss_function):
        self.eval()

        X_tensor = torch.tensor(np.array(X), dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(np.array(y), dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.forward(X_tensor)
            loss = loss_function(logits, y_tensor)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y_tensor).float().mean().item()

        return loss.item(), acc

    def train_model(self, X_train, y_train, X_test, y_test):
        set_seed(self.seed)

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        loss_function = nn.CrossEntropyLoss()

        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int64)
        X_test = np.array(X_test, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.int64)

        best_test_loss = float('inf')
        patience_counter = 0
        best_state = None
        best_epoch = 0

        for epoch in range(self.max_epoch):
            self.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch_X, batch_y in self._get_batches(X_train, y_train, self.batch_size):
                batch_X_tensor = torch.tensor(batch_X, dtype=torch.float32, device=self.device)
                batch_y_tensor = torch.tensor(batch_y, dtype=torch.long, device=self.device)

                optimizer.zero_grad()
                logits = self.forward(batch_X_tensor)
                loss = loss_function(logits, batch_y_tensor)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(batch_X)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_y_tensor).sum().item()
                total += len(batch_X)

            avg_train_loss = epoch_loss / total
            avg_train_acc = correct / total

            test_loss, test_acc = self.evaluate_dataset(X_test, y_test, loss_function)

            self.train_loss_history.append(avg_train_loss)
            self.train_acc_history.append(avg_train_acc)
            self.test_loss_history.append(test_loss)
            self.test_acc_history.append(test_acc)

            print(
                f'Epoch {epoch + 1:03d}/{self.max_epoch} | '
                f'Train Loss: {avg_train_loss:.6f} | Train Acc: {avg_train_acc:.6f} | '
                f'Test Loss: {test_loss:.6f} | Test Acc: {test_acc:.6f}'
            )

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                patience_counter = 0
                best_epoch = epoch + 1
                best_state = copy.deepcopy(self.state_dict())
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f'Early stopping triggered at epoch {epoch + 1}. Best epoch = {best_epoch}.')
                break

        if best_state is not None:
            self.load_state_dict(best_state)

        self.best_epoch = best_epoch

    def test(self, X):
        self.eval()
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits = self.forward(X_tensor)
            pred_y = torch.argmax(logits, dim=1)

        return pred_y.cpu().numpy()

    def run(self):
        print('method running...')
        print('--start training...')

        self.train_model(
            self.data['train']['X'],
            self.data['train']['y'],
            self.data['test']['X'],
            self.data['test']['y']
        )

        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])

        return {
            'pred_y': pred_y,
            'true_y': self.data['test']['y'],
            'train_loss_history': self.train_loss_history,
            'train_acc_history': self.train_acc_history,
            'test_loss_history': self.test_loss_history,
            'test_acc_history': self.test_acc_history,
            'best_epoch': self.best_epoch
        }