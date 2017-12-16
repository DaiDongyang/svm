import load_data


def test_load():
    train_path = './digits/trainingDigits'
    test_path = './digits/testDigits'
    train_set, train_l = load_data.load_sample_set(train_path)
    test_set, test_l = load_data.load_sample_set(test_path)
    print('train:', train_set.shape, train_l.shape)
    print('test:', test_set.shape, test_l.shape)


if __name__=='__main__':
    test_load()
