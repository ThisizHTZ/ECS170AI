from Dataset_Loader import Dataset_Loader
from Method_MLP import Method_MLP
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


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')

    # Dataset
    dataset = Dataset_Loader('stage2 dataset', '')
    dataset.dataset_source_folder_path = script_dir
    dataset.train_file_name = 'train.csv'
    dataset.test_file_name = 'test.csv'
    dataset.normalize = True

    # Method
    method = Method_MLP(
        'mlp',
        '',
        hidden_dims=[256, 128],
        dropout_rate=0.25,
        use_batchnorm=True,
        weight_decay=1e-4,
        patience=5,
        seed=GLOBAL_SEED
    )
    method.max_epoch = 30
    method.learning_rate = 1e-3
    method.batch_size = 128

    # Result saver
    result = Result_Saver('result saver', '')
    result.result_destination_folder_path = results_dir
    result.result_destination_file_name = 'mlp_stage2_result.pkl'

    # Main evaluation used in setting return
    evaluate_acc = Evaluate_Accuracy('accuracy evaluator', '')

    # Setting
    setting = Setting_Train_Test_Split('provided split', '')
    setting.dataset = dataset
    setting.method = method
    setting.result = result
    setting.evaluate = evaluate_acc

    # Run
    accuracy, _ = setting.load_run_save_evaluate()

    # Access saved result directly from result.data
    learned_result = result.data

    # More metrics
    evaluator_precision_macro = Evaluate_Precision('precision macro', '')
    evaluator_precision_macro.average = 'macro'
    evaluator_precision_macro.data = learned_result
    precision_macro = evaluator_precision_macro.evaluate()

    evaluator_recall_macro = Evaluate_Recall('recall macro', '')
    evaluator_recall_macro.average = 'macro'
    evaluator_recall_macro.data = learned_result
    recall_macro = evaluator_recall_macro.evaluate()

    evaluator_f1_macro = Evaluate_F1('f1 macro', '')
    evaluator_f1_macro.average = 'macro'
    evaluator_f1_macro.data = learned_result
    f1_macro = evaluator_f1_macro.evaluate()

    evaluator_precision_weighted = Evaluate_Precision('precision weighted', '')
    evaluator_precision_weighted.average = 'weighted'
    evaluator_precision_weighted.data = learned_result
    precision_weighted = evaluator_precision_weighted.evaluate()

    evaluator_recall_weighted = Evaluate_Recall('recall weighted', '')
    evaluator_recall_weighted.average = 'weighted'
    evaluator_recall_weighted.data = learned_result
    recall_weighted = evaluator_recall_weighted.evaluate()

    evaluator_f1_weighted = Evaluate_F1('f1 weighted', '')
    evaluator_f1_weighted.average = 'weighted'
    evaluator_f1_weighted.data = learned_result
    f1_weighted = evaluator_f1_weighted.evaluate()

    evaluator_precision_micro = Evaluate_Precision('precision micro', '')
    evaluator_precision_micro.average = 'micro'
    evaluator_precision_micro.data = learned_result
    precision_micro = evaluator_precision_micro.evaluate()

    evaluator_recall_micro = Evaluate_Recall('recall micro', '')
    evaluator_recall_micro.average = 'micro'
    evaluator_recall_micro.data = learned_result
    recall_micro = evaluator_recall_micro.evaluate()

    evaluator_f1_micro = Evaluate_F1('f1 micro', '')
    evaluator_f1_micro.average = 'micro'
    evaluator_f1_micro.data = learned_result
    f1_micro = evaluator_f1_micro.evaluate()

    stage2_metrics = {
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

    export_path = export_table(
        [stage2_metrics],
        results_dir,
        'run_stage2_metrics',
        sheet_name='run_stage2'
    )

    print('\n===== Final Test Results =====')
    print(f'Accuracy            : {accuracy:.6f}')
    print(f'Precision (macro)   : {precision_macro:.6f}')
    print(f'Recall (macro)      : {recall_macro:.6f}')
    print(f'F1 (macro)          : {f1_macro:.6f}')
    print(f'Precision (weighted): {precision_weighted:.6f}')
    print(f'Recall (weighted)   : {recall_weighted:.6f}')
    print(f'F1 (weighted)       : {f1_weighted:.6f}')
    print(f'Precision (micro)   : {precision_micro:.6f}')
    print(f'Recall (micro)      : {recall_micro:.6f}')
    print(f'F1 (micro)          : {f1_micro:.6f}')
    if export_path is not None:
        print(f'Exported metrics to : {export_path}')

    # Plot learning curves
    plot_learning_curves(
        learned_result['train_loss_history'],
        learned_result['train_acc_history'],
        learned_result['test_loss_history'],
        learned_result['test_acc_history'],
        results_dir
    )


if __name__ == '__main__':
    main()