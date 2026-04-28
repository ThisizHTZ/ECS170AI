from code.stage_3_code.Dataset_Loader import Dataset_Loader
from code.stage_3_code.Method_CNN import Method_CNN
from code.stage_3_code.Setting_Provided_Split import Setting_Provided_Split
from code.stage_3_code.Result_Saver import Result_Saver
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_3_code.Evaluate_Precision import Evaluate_Precision
from code.stage_3_code.Evaluate_Recall import Evaluate_Recall
from code.stage_3_code.Evaluate_F1 import Evaluate_F1
from code.stage_3_code.Plot_Learning_Curve import plot_learning_curves
import warnings
warnings.filterwarnings("ignore")

def run_one_experiment(dataset_name, exp):
    print(f"Dataset: {dataset_name} | Experiment: {exp['name']}")

    dataset = Dataset_Loader('stage3 dataset', '')
    dataset.dataset_name = dataset_name
    dataset.dataset_source_folder_path = './'

    method = Method_CNN(
        'cnn',
        '',
        dataset_name=dataset_name,
        activation_name=exp['activation_name'],
        loss_name=exp['loss_name'],
        optimizer_name=exp['optimizer_name'],
        pooling_name=exp['pooling_name'],
        dropout_rate=exp['dropout_rate'],
        hidden_dim=exp['hidden_dim'],
        patience=5,
        min_delta=1e-4,
        use_early_stopping=True
    )

    method.max_epoch = 15
    method.learning_rate = exp['learning_rate']
    method.batch_size = 64

    result = Result_Saver('result saver', '')
    result.result_destination_folder_path = './results/'
    result.result_destination_file_name = (
        f"cnn_stage3_{dataset_name.lower()}_{exp['name']}_result.pkl"
    )

    evaluate_acc = Evaluate_Accuracy('accuracy evaluator', '')

    setting = Setting_Provided_Split('provided split', '')
    setting.batch_size = method.batch_size
    setting.dataset = dataset
    setting.method = method
    setting.result = result
    setting.evaluate = evaluate_acc

    accuracy, _ = setting.load_run_save_evaluate()
    learned_result = result.data

    learned_result['pred_y'] = [int(x) for x in learned_result['pred_y']]
    learned_result['true_y'] = [int(x) for x in learned_result['true_y']]

    evaluator_precision = Evaluate_Precision('precision macro', '')
    evaluator_precision.average = 'macro'
    evaluator_precision.data = learned_result
    precision_macro = evaluator_precision.evaluate()

    evaluator_recall = Evaluate_Recall('recall macro', '')
    evaluator_recall.average = 'macro'
    evaluator_recall.data = learned_result
    recall_macro = evaluator_recall.evaluate()

    evaluator_f1 = Evaluate_F1('f1 macro', '')
    evaluator_f1.average = 'macro'
    evaluator_f1.data = learned_result
    f1_macro = evaluator_f1.evaluate()

    print('\n===== Final Test Results =====')
    print(f"Dataset          : {dataset_name}")
    print(f"Experiment       : {exp['name']}")
    print(f"Accuracy         : {accuracy:.6f}")
    print(f"Precision (macro): {precision_macro:.6f}")
    print(f"Recall (macro)   : {recall_macro:.6f}")
    print(f"F1 (macro)       : {f1_macro:.6f}")

    plot_learning_curves(
        learned_result['train_loss_history'],
        learned_result['train_acc_history'],
        './results/',
        dataset_name,
        exp['name']
    )

    return {
        'dataset': dataset_name,
        'experiment': exp['name'],
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro
    }


def main():
    dataset_names = ['MNIST', 'CIFAR', 'ORL']

    experiments = [
        {
            'name': 'baseline',
            'activation_name': 'relu',
            'loss_name': 'cross_entropy',
            'optimizer_name': 'adam',
            'pooling_name': 'max',
            'dropout_rate': 0.2,
            'hidden_dim': 128,
            'learning_rate': 1e-3
        },
        {
            'name': 'leaky_relu',
            'activation_name': 'leaky_relu',
            'loss_name': 'cross_entropy',
            'optimizer_name': 'adam',
            'pooling_name': 'max',
            'dropout_rate': 0.2,
            'hidden_dim': 128,
            'learning_rate': 1e-3
        },
        {
            'name': 'label_smoothing',
            'activation_name': 'relu',
            'loss_name': 'label_smoothing',
            'optimizer_name': 'adam',
            'pooling_name': 'max',
            'dropout_rate': 0.2,
            'hidden_dim': 128,
            'learning_rate': 1e-3
        },
        {
            'name': 'adamw',
            'activation_name': 'relu',
            'loss_name': 'cross_entropy',
            'optimizer_name': 'adamw',
            'pooling_name': 'max',
            'dropout_rate': 0.2,
            'hidden_dim': 128,
            'learning_rate': 1e-3
        },
        {
            'name': 'avg_pool',
            'activation_name': 'relu',
            'loss_name': 'cross_entropy',
            'optimizer_name': 'adam',
            'pooling_name': 'avg',
            'dropout_rate': 0.2,
            'hidden_dim': 128,
            'learning_rate': 1e-3
        },
        {
            'name': 'larger_hidden',
            'activation_name': 'relu',
            'loss_name': 'cross_entropy',
            'optimizer_name': 'adam',
            'pooling_name': 'max',
            'dropout_rate': 0.2,
            'hidden_dim': 256,
            'learning_rate': 1e-3
        }
    ]

    all_results = []

    for dataset_name in dataset_names:
        for exp in experiments:
            result = run_one_experiment(dataset_name, exp)
            all_results.append(result)

    print("Summary of All Experiments")

    for r in all_results:
        print(
            f"{r['dataset']:6s} | "
            f"{r['experiment']:15s} | "
            f"Acc: {r['accuracy']:.6f} | "
            f"Precision: {r['precision_macro']:.6f} | "
            f"Recall: {r['recall_macro']:.6f} | "
            f"F1: {r['f1_macro']:.6f}"
        )

if __name__ == '__main__':
    main()
'''
def main():
    dataset_name = 'ORL'   # 'MNIST' / 'CIFAR' / 'ORL'

    dataset = Dataset_Loader('stage3 dataset', '')
    dataset.dataset_name = dataset_name
    dataset.dataset_source_folder_path = './'   # 如果文件不在data文件夹，就改成 './'

    method = Method_CNN('cnn', '', dataset_name=dataset_name)
    method.max_epoch = 30
    method.learning_rate = 1e-3
    method.batch_size = 64

    result = Result_Saver('result saver', '')
    result.result_destination_folder_path = './results/'
    result.result_destination_file_name = f'cnn_stage3_{dataset_name.lower()}_result.pkl'

    evaluate_acc = Evaluate_Accuracy('accuracy evaluator', '')

    setting = Setting_Provided_Split('provided split', '')
    setting.batch_size = method.batch_size
    setting.dataset = dataset
    setting.method = method
    setting.result = result
    setting.evaluate = evaluate_acc

    accuracy, _ = setting.load_run_save_evaluate()
    learned_result = result.data

    evaluator_precision = Evaluate_Precision('precision macro', '')
    evaluator_precision.average = 'macro'
    evaluator_precision.data = learned_result
    precision_macro = evaluator_precision.evaluate()

    evaluator_recall = Evaluate_Recall('recall macro', '')
    evaluator_recall.average = 'macro'
    evaluator_recall.data = learned_result
    recall_macro = evaluator_recall.evaluate()

    evaluator_f1 = Evaluate_F1('f1 macro', '')
    evaluator_f1.average = 'macro'
    evaluator_f1.data = learned_result
    f1_macro = evaluator_f1.evaluate()

    print('\n===== Final Test Results =====')
    print(f'Dataset          : {dataset_name}')
    print(f'Accuracy         : {accuracy:.6f}')
    print(f'Precision (macro): {precision_macro:.6f}')
    print(f'Recall (macro)   : {recall_macro:.6f}')
    print(f'F1 (macro)       : {f1_macro:.6f}')

    plot_learning_curves(
        learned_result['train_loss_history'],
        learned_result['train_acc_history'],
        './results/'
    )
if __name__ == '__main__':
    main()
'''