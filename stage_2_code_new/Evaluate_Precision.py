'''
Precision evaluator for multiclass classification
'''

from Base_Classes import evaluate
from sklearn.metrics import precision_score


class Evaluate_Precision(evaluate):
    data = None
    average = 'macro'

    def evaluate(self):
        print(f'evaluating precision ({self.average})...')
        return precision_score(
            self.data['true_y'],
            self.data['pred_y'],
            average=self.average,
            zero_division=0
        )