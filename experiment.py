import my_svm
import main
import config
import load_data
import numpy as np
import time
import gc

log_level = config.log_level


def run(C, k_tup, is_simple):
    train_dir = config.train_path
    test_dir = config.test_path
    classes = config.classes
    train_data, train_ls = load_data.load_sample_set(train_dir)
    test_data, test_ls = load_data.load_sample_set(test_dir)
    toler = config.toler
    max_iter = config.max_iter
    t1 = time.time()
    models = main.train_models(train_dir, classes, C, toler, max_iter, k_tup, is_simple)
    t2 = time.time()

    #  train set
    pr_2d_list0 = list()
    for model_tuple in models:
        if log_level > 0:
            print('training set test between %d and %d' % (model_tuple[1], model_tuple[2]))
        result0 = main.model_predict(model_tuple, train_data)
        pr_2d_list0.append(result0)
    pr_mat0 = np.array(pr_2d_list0)
    results0 = main.prmat2result(pr_mat0, classes)
    check0 = (results0 == train_ls)
    acc0 = np.sum(check0) / len(check0)
    t3 = time.time()

    #  test set
    pr_2d_list1 = list()
    for model_tuple in models:
        if log_level > 0:
            print('test set test between %d and %d' % (model_tuple[1], model_tuple[2]))
        result1 = main.model_predict(model_tuple, test_data)
        pr_2d_list1.append(result1)
    pr_mat1 = np.array(pr_2d_list1)
    results1 = main.prmat2result(pr_mat1, classes)
    check1 = (results1 == test_ls)
    acc1 = np.sum(check1) / len(check1)
    t4 = time.time()

    if is_simple:
        print('simple smo')
    else:
        print('complete smo')
    print('C:', C)
    print('k_tup:', k_tup)
    print('training time: %.3fs' % (t2 - t1))
    print('training set test time: %.3fs' % (t3 - t2))
    print('training set accuracy:', acc0)
    print('training set error rate:', 1 - acc0)
    print('test set test time: %.3fs' % (t4 - t3))
    print('test set accuracy:', acc1)
    print('test set error rate:', 1 - acc1)
    print()
    # np.save('results', results)
    # np.save('test_ls', test_ls)
    del train_data, train_ls, test_data, test_ls, pr_2d_list0, pr_2d_list1, pr_mat0, pr_mat1
    gc.collect()


if __name__ == '__main__':
    t1 = time.time()
    Cs = [1e-1, 1, 10]
    gammas = [1e-3, 1e-2, 1e-1, 1, 10]
    # Cs = [0.1]
    # gammas = [1e-8]
    is_simples = [False, True]
    # lin
    for is_simple in is_simples:
        for C in Cs:
            # line
            k_tup = ('lin', 1)
            run(C, k_tup, is_simple)

    #       # rbf
            for gamma in gammas:
                k_tup = ('rbf', gamma)
                run(C, k_tup, is_simple)
    t2 = time.time()
    print('total time %.3fs' % (t2 - t1))

