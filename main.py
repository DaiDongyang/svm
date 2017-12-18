import my_svm
import config
import load_data
import numpy as np
import time

log_level = config.log_level


def train_models(train_dir, classes, C, toler, max_iter, k_tup, is_simple):
    train_set, train_l = load_data.load_sample_set(train_dir)
    subsets = load_data.get_subsets(train_set, train_l, classes)
    classes_num = len(classes)
    models = list()
    for i in range(classes_num):
        for j in range(i+1, classes_num):
            if log_level > 0:
                print('train between %d and %d' % (classes[i], classes[j]))
            model = train_2c(subsets, i, j, C, toler, max_iter, k_tup, is_simple)
            models.append((model, i, j))
    return models


def train_2c(subset, i, j, C, toler, max_iter, k_tup, is_simple):
    datai = subset[i]
    dataj = subset[j]
    lsi = np.ones((len(datai), 1))
    lsj = np.ones((len(dataj), 1)) * (-1)
    data = np.vstack((datai, dataj))
    ls = np.vstack((lsi, lsj))
    if is_simple:
        model = my_svm.sim_smo(data, ls, C, toler, max_iter, k_tup)
    else:
        model = my_svm.smo(data, ls, C, toler, max_iter, k_tup)
    return model


def model_predict(model_tuple, data):
    model, i, j = model_tuple
    results_origin = my_svm.predict_origin(model, data)
    # the element in classes should be number
    results = np.zeros(results_origin.shape, dtype=int)
    results[results_origin >= 0] = i
    results[results_origin < 0] = j
    return list(results.flat)


def prmat2result(pr_mat, classes):
    _, n = pr_mat.shape
    freq_mat = np.zeros((len(classes), n))
    n_idx = np.array(range(n))
    for row in pr_mat:
        freq_mat[row, n_idx] += 1
    r_idx = np.argmax(freq_mat, axis=0)
    classes_np = np.array(classes)
    return classes_np[r_idx]


def main():
    train_dir = config.train_path
    test_dir = config.test_path
    classes = config.classes
    C = config.C
    toler = config.toler
    max_iter = config.max_iter
    k_tup = config.k_tup
    is_simple = config.is_simple
    test_data, test_ls = load_data.load_sample_set(test_dir)
    t1 = time.time()
    models = train_models(train_dir, classes, C, toler, max_iter, k_tup, is_simple)
    t2 = time.time()
    pr_2d_list = list()
    for model_tuple in models:
        if log_level > 0:
            print('test between %d and %d' % (model_tuple[1], model_tuple[2]))
        result = model_predict(model_tuple, test_data)
        pr_2d_list.append(result)
    pr_mat = np.array(pr_2d_list)
    results = prmat2result(pr_mat, classes)
    t3 = time.time()
    check = (results == test_ls)
    acc = np.sum(check) / len(check)
    print('k_tup:', k_tup)
    if is_simple:
        print('simple smo')
    else:
        print('complete smo')
    print('training time:', t2 - t1)
    print('test time:', t3 - t2)
    print('accuracy:', acc)
    # np.save('results', results)
    # np.save('test_ls', test_ls)


if __name__ == '__main__':
    main()



