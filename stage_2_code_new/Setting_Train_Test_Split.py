from Base_Classes import setting

class Setting_Train_Test_Split(setting):

    def load_run_save_evaluate(self):

        train_data = self.dataset.load_train()
        test_data = self.dataset.load_test()

        self.method.data = {
            'train': train_data,
            'test': test_data
        }

        learned_result = self.method.run()

        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate(), None