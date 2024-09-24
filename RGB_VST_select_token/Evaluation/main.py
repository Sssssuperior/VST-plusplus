import os.path as osp
from .evaluator import Eval_thread
from .dataloader import EvalDataset


def evaluate(args):

    pred_dir = args.save_test_path_root
    output_dir = args.save_dir
    gt_dir = args.data_root

    method_names1 = args.methods1.split('+')
    method_names2 = args.methods2.split('+')

    threads = []
    test_paths = args.test_paths.split('+')
    for dataset_setname in test_paths:

        dataset_name = dataset_setname.split('/')[0]

        for method1 in method_names1:

            pred_dir_all = osp.join(pred_dir, dataset_name, method1)

            if dataset_name == 'DUTS':
                gt_dir_all = osp.join(osp.join(gt_dir, dataset_setname)) + '/DUTS-TE-Mask'
            else:
                gt_dir_all = osp.join(osp.join(gt_dir, dataset_setname)) + '/GT'

            loader = EvalDataset(pred_dir_all, gt_dir_all)
            thread = Eval_thread(loader, method1, dataset_setname, output_dir, cuda=True)
            threads.append(thread)
        for method2 in method_names2:

            pred_dir_all = osp.join(pred_dir, dataset_name, method2)

            if dataset_name == 'DUTS':
                gt_dir_all = osp.join(osp.join(gt_dir, dataset_setname)) + '/DUTS-TE-Mask'
            else:
                gt_dir_all = osp.join(osp.join(gt_dir, dataset_setname)) + '/GT'

            loader = EvalDataset(pred_dir_all, gt_dir_all)
            thread = Eval_thread(loader, method2, dataset_setname, output_dir, cuda=True)
            threads.append(thread)
    for thread in threads:
        print(thread.run())
