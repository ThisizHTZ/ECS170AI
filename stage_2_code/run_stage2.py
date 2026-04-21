from code.stage_2_code.Dataset_Loader import Dataset_Loader
from code.stage_2_code.Method_MLP import Method_MLP
from code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from code.stage_2_code.Result_Saver import Result_Saver
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.stage_2_code.Evaluate_Precision import Evaluate_Precision
from code.stage_2_code.Evaluate_Recall import Evaluate_Recall
from code.stage_2_code.Evaluate_F1 import Evaluate_F1
from code.stage_2_code.Plot_Learning_Curve import plot_learning_curves


def main():
    # Dataset
    dataset = Dataset_Loader('stage2 dataset', '')
    dataset.dataset_source_folder_path = './'
    dataset.train_file_name = 'train.csv'
    dataset.test_file_name = 'test.csv'
    dataset.normalize = True

    # Method
    method = Method_MLP('mlp', '')
    method.max_epoch = 30
    method.learning_rate = 1e-3
    method.batch_size = 128

    # Result saver
    result = Result_Saver('result saver', '')
    result.result_destination_folder_path = './results/'
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

    print('\n===== Final Test Results =====')
    print(f'Accuracy            : {accuracy:.6f}')
    print(f'Precision (macro)   : {precision_macro:.6f}')
    print(f'Recall (macro)      : {recall_macro:.6f}')
    print(f'F1 (macro)          : {f1_macro:.6f}')
    print(f'Precision (weighted): {precision_weighted:.6f}')
    print(f'Recall (weighted)   : {recall_weighted:.6f}')
    print(f'F1 (weighted)       : {f1_weighted:.6f}')

    # Plot learning curves
    plot_learning_curves(
        learned_result['train_loss_history'],
        learned_result['train_acc_history'],
        learned_result['test_loss_history'],
        learned_result['test_acc_history'],
        './results/'
    )


if __name__ == '__main__':
    main()