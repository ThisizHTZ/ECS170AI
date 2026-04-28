from code.base_class.setting import setting
from torch.utils.data import DataLoader, TensorDataset
import torch


class Setting_Provided_Split(setting):
    batch_size = 64

    def load_run_save_evaluate(self):
        train_data = self.dataset.load_train()
        test_data = self.dataset.load_test()

        train_dataset = TensorDataset(
            torch.tensor(train_data['X'], dtype=torch.float32),
            torch.tensor(train_data['y'], dtype=torch.long)
        )

        test_dataset = TensorDataset(
            torch.tensor(test_data['X'], dtype=torch.float32),
            torch.tensor(test_data['y'], dtype=torch.long)
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        self.method.data = {
            'train_loader': train_loader,
            'test_loader': test_loader
        }

        learned_result = self.method.run()

        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result
        score = self.evaluate.evaluate()

        return score, None