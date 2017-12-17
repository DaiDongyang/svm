import load_data
import config


def test_load():
    train_path = './digits/trainingDigits'
    test_path = './digits/testDigits'
    train_set, train_l = load_data.load_sample_set(train_path)
    test_set, test_l = load_data.load_sample_set(test_path)
    print('train:', train_set.shape, train_l.shape)
    print('test:', test_set.shape, test_l.shape)
    # load_data.extract_subset(train_set, train_l, 1, 2)
    # trains = list()
    # lss = list()
    # num = 0
    # for i in config.classes:
    #     sub_train, sub_l = load_data.extract_subset(train_set, train_l, i)
    #     print(sub_train.shape, sub_l.shape)
    #     num += len(sub_l)
    # print(num)
    subsets = load_data.get_subsets(train_set, train_l, config.classes)
    for i in subsets:
        print(i.shape)


if __name__=='__main__':
    test_load()
