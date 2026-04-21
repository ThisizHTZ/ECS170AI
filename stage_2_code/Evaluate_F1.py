'''
F1 evaluator for multiclass classification
'''

from code.base_class.evaluate import evaluate
from sklearn.metrics import f1_score


class Evaluate_F1(evaluate):
    data = None
    average = 'macro'

    def evaluate(self):
        print(f'evaluating f1 ({self.average})...')
        return f1_score(
            self.data['true_y'],
            self.data['pred_y'],
            average=self.average,
            zero_division=0
        )