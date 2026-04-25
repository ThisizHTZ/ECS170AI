'''
Recall evaluator for multiclass classification
'''

from Base_Classes import evaluate
from sklearn.metrics import recall_score


class Evaluate_Recall(evaluate):
    data = None
    average = 'macro'

    def evaluate(self):
        print(f'evaluating recall ({self.average})...')
        return recall_score(
            self.data['true_y'],
            self.data['pred_y'],
            average=self.average,
            zero_division=0
        )