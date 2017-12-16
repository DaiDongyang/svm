import numpy as np
import random


def rbf_trans(data1, data2, gamma):
    """
    return a matrix of kernel result.
    :param data1: m1 * d array, in order to improve efficiency, let m1 <= m2
    :param data2: m2 * d array
    :param gamma:K(xi, xj) = exp(-gamma (xi - xj) **2)
    :return: return a matrix K, K[i, j] means K(xi, xj)
    """
    m1, d1 = np.shape(data1)
    m2, d2 = np.shape(data2)
    d = d1
    if d != d2:
        raise NameError('dimension not equal')
    K = np.zeros((m1, m2))
    for i in range(m1):
        diff_sq = np.sum((data1[i, :] - data2) ** 2, axis=1)
        rbf = np.exp(-1 * gamma * diff_sq)
        K[i, :] = rbf.T
    return K


def line_trans(data1, data2):
    """
    return a matrix of kernel result.
    :param data1: m1 * d array, in order to improve efficiency, let m1 <= m2
    :param data2: m2 * d array
    :return: return a matrix K, K[i, j] means K(xi, xj) = np.dot(xi, xj)
    """
    K = np.dot(data1, data2.T)
    return K


def select_j_rand(i, m):
    j = i
    while j == i:
        j = random.randint(0, m - 1)
    return j


class Meta:
    def __init__(self, data, ls, C, toler, k_tup):
        self.b = 0
        self.X = data
        self.C = C
        self.tol = toler
        self.k_tup = k_tup
        self.m = np.shape(data)[0]
        self.y = ls.reshape((-1, 1))
        # self.a is a array
        self.a = np.zeros((self.m, 1))
        self.e_cache = np.zeros((self.m, 1))
        self.K = np.zeros((self.m, self.m))
        if k_tup[0] == 'line' or k_tup[0] == 'lin':
            self.K = line_trans(data, data)
        elif k_tup[0] == 'rbf':
            self.K = rbf_trans(data, data, k_tup[1])
        else:
            raise NameError("The kernel is not recognized, please make sure kernel is 'rbf' or 'line'")
        self.calc_e()

    def calc_e(self):
        gx = np.dot(self.K, self.y * self.a) + self.b
        self.e_cache = gx - self.y

    def update_ei(self, i):
        gxi = np.dot(self.K[i, :], self.y * self.a) + self.b
        self.e_cache[i] = gxi - self.y[i]

    def calc_e2(self):
        return np.dot(self.K, self.y * self.a) + self.b - self.y


def select_j(i, meta, is_random):
    ei = meta.e_cache[i]
    delta_e = np.abs(meta.e_cache - ei)
    if is_random:
        j = np.random.choice(np.flatnonzero(delta_e == delta_e.max()))
    else:
        j = np.argmax(delta_e)
    if j == i:
        j = select_j_rand(i, meta.m)
    return j, float(meta.e_cache[j])


def inner_l(i, meta, is_simple):
    ei = float(meta.e_cache[i])
    # kkt check
    if (float(meta.y[i]) * ei < -meta.tol and float(meta.a[i]) < meta.C) or (
            float(meta.y[i]) * ei > meta.tol and float(meta.a[i]) > 0):
        if is_simple:
            j = select_j_rand(i, meta.m)
            ej = float(meta.e_cache[j])
        else:
            j, ej = select_j(i, meta, False)
        # print('meta.a\n', meta.a)
        ai_old = float(meta.a[i])
        aj_old = float(meta.a[j])
        if meta.y[i] != meta.y[j]:
            low = max(0.0, aj_old - ai_old)
            high = min(meta.C, meta.C + aj_old - ai_old)
        else:
            low = max(0.0, ai_old + aj_old - meta.C)
            high = min(meta.C, ai_old + aj_old)
        if low == high:
            print('l == h')
            return 0
        # print('L, H, alphaiold', low, high, ai_old)
        eta = 2.0 * meta.K[i, j] - meta.K[i, i] - meta.K[j, j]
        if eta >= 0:
            print('eta >= 0')
            return 0
        aj = aj_old - meta.y[j] * (ei - ej) / eta
        aj = np.clip(aj, low, high)
        if np.abs(aj - aj_old) < 0.00001:
            print('j not moving enough')
            return 0
        meta.a[j] = aj
        meta.a[i] += meta.y[j] * meta.y[i] * (aj_old - meta.a[j])  # update i by the same amount as j
        b1 = meta.b - ei - meta.y[i] * (meta.a[i] - ai_old) * meta.K[i, i] - meta.y[j] * (meta.a[j] - aj_old) * meta.K[
            i, j]
        b2 = meta.b - ej - meta.y[i] * (meta.a[i] - ai_old) * meta.K[i, j] - meta.y[j] * (meta.a[j] - aj_old) * meta.K[
            j, j]
        if 0 < meta.a[i] < meta.C:
            meta.b = b1
        elif 0 < meta.a[j] < meta.C:
            meta.b = b2
        else:
            meta.b = (b1 + b2) / 2
        meta.calc_e()
        # print(meta.a)
        return 1
    return 0


def smo(data, ls, C, toler, max_iter, k_tup=('rbf', 1)):
    is_simple = False
    meta = Meta(data, ls, C, toler, k_tup)
    # iteration
    it = 0
    entire_set = False
    pair_changed = 1
    while it < max_iter and (entire_set or pair_changed > 0):
        pair_changed = 0
        if entire_set:
            for i in range(meta.m):
                pair_changed += inner_l(i, meta, is_simple)
                print('fullSet, iter: %d i:%d, pair changed %d' % (it, i, pair_changed))
        else:
            non_bound_idxes = np.nonzero((meta.a > 0) * (meta.a < meta.C))[0]
            for i in non_bound_idxes:
                pair_changed += inner_l(i, meta, is_simple)
                print('non-bound, iter: %d i:%d, pair changed %d' % (it, i, pair_changed))
        it += 1
        if entire_set:
            entire_set = False
        elif pair_changed == 0:
            entire_set = True
        print('iter number: %d' % it)
    return meta


def sim_smo(data, ls, C, toler, max_iter, k_tup=('rbf', 1)):
    is_simple = True
    meta = Meta(data, ls, C, toler, k_tup)
    # iteration
    it = 0
    pair_change = 1
    while it < max_iter and pair_change > 0:
        pair_change = 0
        for i in range(meta.m):
            pair_change += inner_l(i, meta, is_simple)
            print('simple smo, iter: %d, i%d,pair changed %d' % (it, i, pair_change))
        it += 1
        print('iter number: %d' % it)
    return meta


def predict(meta, data):
    a = meta.a
    sv_ind = np.nonzero(a > 0)[0]
    sv_x = meta.X[sv_ind]
    sv_y = meta.y[sv_ind]
    sv_a = meta.a[sv_ind]
    # if meta.k_tup[0] == 'lin' or meta.k_tup[0] == 'line':
    #     K = line_trans(sv_x, data)
    if meta.k_tup[0] == 'rbf':
        K = rbf_trans(sv_x, data, meta.k_tup[1])
    else:
        K = line_trans(sv_x, data)
    results = (np.dot((sv_y * sv_a).T, K)).T + meta.b
    return results

