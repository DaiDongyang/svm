import main
import config
import numpy as np


def test_pr_mat2result():
    pr_mat = np.load('./pr_mat.npy')
    test_ls = np.load('./test_ls.npy')
    results = main.pr_mat2result(pr_mat, config.classes)
    check = test_ls == results
    print(np.sum(check))
    print(len(check))


if __name__ == '__main__':
    test_pr_mat2result()
