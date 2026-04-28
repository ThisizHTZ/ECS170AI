from code.base_class.method import method
import torch
from torch import nn

class Method_CNN(method, nn.Module):
    data = None
    max_epoch = 15
    learning_rate = 1e-3
    batch_size = 64
    device = None

    def __init__(
        self,
        mName,
        mDescription,
        dataset_name='ORL',
        activation_name='relu',
        loss_name='cross_entropy',
        optimizer_name='adam',
        pooling_name='max',
        kernel_size=3,
        padding=1,
        dropout_rate=0.2,
        hidden_dim=128,
        patience=5,
        min_delta=1e-4,
        use_early_stopping=True
    ):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.dataset_name = dataset_name
        self.activation_name = activation_name
        self.loss_name = loss_name
        self.optimizer_name = optimizer_name
        self.pooling_name = pooling_name
        self.kernel_size = kernel_size
        self.padding = padding
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.patience = patience
        self.min_delta = min_delta
        self.use_early_stopping = use_early_stopping

        if self.dataset_name == 'MNIST':
            in_channels = 1
            num_classes = 10
            H, W = 28, 28

        elif self.dataset_name == 'CIFAR':
            in_channels = 3
            num_classes = 10
            H, W = 32, 32

        elif self.dataset_name == 'ORL':
            in_channels = 1
            num_classes = 40
            H, W = 112, 92

        else:
            raise ValueError('Unsupported dataset_name')

        activation = self.get_activation()
        pooling = self.get_pooling()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=self.kernel_size, padding=self.padding),
            activation,
            pooling,

            nn.Conv2d(32, 64, kernel_size=self.kernel_size, padding=self.padding),
            self.get_activation(),
            self.get_pooling()
        )

        flattened_dim = self.get_flattened_dim(in_channels, H, W)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, self.hidden_dim),
            self.get_activation(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, num_classes)
        )

        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

        self.train_loss_history = []
        self.train_acc_history = []

    def get_activation(self):
        if self.activation_name == 'relu':
            return nn.ReLU()
        elif self.activation_name == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif self.activation_name == 'tanh':
            return nn.Tanh()
        elif self.activation_name == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError('Unsupported activation_name')

    def get_pooling(self):
        if self.pooling_name == 'max':
            return nn.MaxPool2d(kernel_size=2)
        elif self.pooling_name == 'avg':
            return nn.AvgPool2d(kernel_size=2)
        else:
            raise ValueError('Unsupported pooling_name')

    def get_loss_function(self):
        if self.loss_name == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif self.loss_name == 'label_smoothing':
            return nn.CrossEntropyLoss(label_smoothing=0.1)
        else:
            raise ValueError('Unsupported loss_name')

    def get_optimizer(self):
        if self.optimizer_name == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == 'adamw':
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        elif self.optimizer_name == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            raise ValueError('Unsupported optimizer_name')

    def get_flattened_dim(self, in_channels, H, W):
        with torch.no_grad():
            x = torch.zeros(1, in_channels, H, W)
            x = self.features(x)
            return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def train_model(self, train_loader):
        optimizer = self.get_optimizer()
        loss_function = self.get_loss_function()

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.max_epoch):
            self.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).long()

                logits = self.forward(images)
                loss = loss_function(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * images.size(0)

                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            avg_loss = epoch_loss / total
            avg_acc = correct / total

            self.train_loss_history.append(avg_loss)
            self.train_acc_history.append(avg_acc)

            print(
                f'Epoch {epoch + 1}/{self.max_epoch} | '
                f'Loss: {avg_loss:.6f} | '
                f'Accuracy: {avg_acc:.6f}'
            )

            if self.use_early_stopping:
                if avg_loss < best_loss - self.min_delta:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break

    def test(self, test_loader):
        self.eval()
        pred_y = []
        true_y = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).long()

                logits = self.forward(images)
                preds = torch.argmax(logits, dim=1)

                pred_y.extend(preds.cpu().numpy().astype(int).tolist())
                true_y.extend(labels.cpu().numpy().astype(int).tolist())

        return pred_y, true_y

    def run(self):
        print('method running...')
        print('--start training...')
        self.train_model(self.data['train_loader'])

        print('--start testing...')
        pred_y, true_y = self.test(self.data['test_loader'])

        return {
            'pred_y': pred_y,
            'true_y': true_y,
            'train_loss_history': self.train_loss_history,
            'train_acc_history': self.train_acc_history
        }