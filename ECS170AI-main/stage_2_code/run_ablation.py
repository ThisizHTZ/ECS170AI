from Dataset_Loader import Dataset_Loader
from Method_MLP import Method_MLP, set_seed
from Setting_Train_Test_Split import Setting_Train_Test_Split
from Result_Saver import Result_Saver
from Evaluate_Accuracy import Evaluate_Accuracy
from Evaluate_Precision import Evaluate_Precision
from Evaluate_Recall import Evaluate_Recall
from Evaluate_F1 import Evaluate_F1
from Plot_Learning_Curve import plot_learning_curves
from Export_Results import export_table
import os


GLOBAL_SEED = 42


def evaluate_all_metrics(learned_result):
    accuracy_evaluator = Evaluate_Accuracy('accuracy', '')
    accuracy_evaluator.data = learned_result
    accuracy = accuracy_evaluator.evaluate()

    precision_macro_evaluator = Evaluate_Precision('precision macro', '')
    precision_macro_evaluator.average = 'macro'
    precision_macro_evaluator.data = learned_result
    precision_macro = precision_macro_evaluator.evaluate()

    recall_macro_evaluator = Evaluate_Recall('recall macro', '')
    recall_macro_evaluator.average = 'macro'
    recall_macro_evaluator.data = learned_result
    recall_macro = recall_macro_evaluator.evaluate()

    f1_macro_evaluator = Evaluate_F1('f1 macro', '')
    f1_macro_evaluator.average = 'macro'
    f1_macro_evaluator.data = learned_result
    f1_macro = f1_macro_evaluator.evaluate()

    precision_weighted_evaluator = Evaluate_Precision('precision weighted', '')
    precision_weighted_evaluator.average = 'weighted'
    precision_weighted_evaluator.data = learned_result
    precision_weighted = precision_weighted_evaluator.evaluate()

    recall_weighted_evaluator = Evaluate_Recall('recall weighted', '')
    recall_weighted_evaluator.average = 'weighted'
    recall_weighted_evaluator.data = learned_result
    recall_weighted = recall_weighted_evaluator.evaluate()

    f1_weighted_evaluator = Evaluate_F1('f1 weighted', '')
    f1_weighted_evaluator.average = 'weighted'
    f1_weighted_evaluator.data = learned_result
    f1_weighted = f1_weighted_evaluator.evaluate()

    precision_micro_evaluator = Evaluate_Precision('precision micro', '')
    precision_micro_evaluator.average = 'micro'
    precision_micro_evaluator.data = learned_result
    precision_micro = precision_micro_evaluator.evaluate()

    recall_micro_evaluator = Evaluate_Recall('recall micro', '')
    recall_micro_evaluator.average = 'micro'
    recall_micro_evaluator.data = learned_result
    recall_micro = recall_micro_evaluator.evaluate()

    f1_micro_evaluator = Evaluate_F1('f1 micro', '')
    f1_micro_evaluator.average = 'micro'
    f1_micro_evaluator.data = learned_result
    f1_micro = f1_micro_evaluator.evaluate()

    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro
    }


def run_experiment(exp_name, hidden_dims, dropout_rate, use_batchnorm, weight_decay, seed=42):
    print(f'\n===== Running {exp_name} =====')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')

    set_seed(seed)

    dataset = Dataset_Loader('stage2 dataset', '')
    dataset.dataset_source_folder_path = script_dir
    dataset.train_file_name = 'train.csv'
    dataset.test_file_name = 'test.csv'
    dataset.normalize = True

    method = Method_MLP(
        exp_name,
        '',
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        use_batchnorm=use_batchnorm,
        weight_decay=weight_decay,
        patience=5,
        seed=seed
    )
    method.max_epoch = 30
    method.learning_rate = 1e-3
    method.batch_size = 128

    result = Result_Saver('result saver', '')
    result.result_destination_folder_path = results_dir
    result.result_destination_file_name = f'{exp_name}.pkl'

    evaluate_acc = Evaluate_Accuracy('accuracy evaluator', '')

    setting = Setting_Train_Test_Split('provided split', '')
    setting.dataset = dataset
    setting.method = method
    setting.result = result
    setting.evaluate = evaluate_acc

    accuracy, _ = setting.load_run_save_evaluate()
    learned_result = result.data
    metrics = evaluate_all_metrics(learned_result)

    return {
        'experiment': exp_name,
        'architecture': str(hidden_dims),
        'dropout': dropout_rate,
        'batchnorm': use_batchnorm,
        'weight_decay': weight_decay,
        'best_epoch': learned_result.get('best_epoch', None),
        'accuracy': metrics['accuracy'],
        'precision_macro': metrics['precision_macro'],
        'recall_macro': metrics['recall_macro'],
        'f1_macro': metrics['f1_macro'],
        'precision_weighted': metrics['precision_weighted'],
        'recall_weighted': metrics['recall_weighted'],
        'f1_weighted': metrics['f1_weighted'],
        'precision_micro': metrics['precision_micro'],
        'recall_micro': metrics['recall_micro'],
        'f1_micro': metrics['f1_micro']
    }


def main():
    experiments = [
        {
            'exp_name': 'baseline',
            'hidden_dims': [256, 128],
            'dropout_rate': 0.25,
            'use_batchnorm': True,
            'weight_decay': 1e-4
        },
        {
            'exp_name': 'deeper_model',
            'hidden_dims': [256, 128, 64],
            'dropout_rate': 0.25,
            'use_batchnorm': True,
            'weight_decay': 1e-4
        },
        {
            'exp_name': 'smaller_model',
            'hidden_dims': [128, 64],
            'dropout_rate': 0.25,
            'use_batchnorm': True,
            'weight_decay': 1e-4
        },
        {
            'exp_name': 'no_dropout',
            'hidden_dims': [256, 128],
            'dropout_rate': 0.0,
            'use_batchnorm': True,
            'weight_decay': 1e-4
        },
        {
            'exp_name': 'no_batchnorm',
            'hidden_dims': [256, 128],
            'dropout_rate': 0.25,
            'use_batchnorm': False,
            'weight_decay': 1e-4
        },
        {
            'exp_name': 'no_weight_decay',
            'hidden_dims': [256, 128],
            'dropout_rate': 0.25,
            'use_batchnorm': True,
            'weight_decay': 0.0
        }
    ]

    all_results = []

    for exp in experiments:
        result = run_experiment(
            exp_name=exp['exp_name'],
            hidden_dims=exp['hidden_dims'],
            dropout_rate=exp['dropout_rate'],
            use_batchnorm=exp['use_batchnorm'],
            weight_decay=exp['weight_decay'],
            seed=GLOBAL_SEED
        )
        all_results.append(result)

    print('\n===== Ablation Study Summary =====')
    for r in all_results:
        print(
            f"{r['experiment']:15s} | "
            f"arch={r['architecture']:15s} | "
            f"dropout={r['dropout']:<4} | "
            f"BN={r['batchnorm']} | "
            f"wd={r['weight_decay']:.0e} | "
            f"best_epoch={r['best_epoch']} | "
            f"acc={r['accuracy']:.6f} | "
            f"macro_f1={r['f1_macro']:.6f} | "
            f"weighted_f1={r['f1_weighted']:.6f} | "
            f"micro_f1={r['f1_micro']:.6f}"
        )

    export_path = export_table(
        all_results,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'),
        'run_ablation_metrics',
        sheet_name='ablation'
    )
    if export_path is not None:
        print(f'Exported metrics to : {export_path}')


if __name__ == '__main__':
    main()