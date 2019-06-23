import csv
import utils

train_file_name = "data/train.csv"

def read_data(file_name, limit = 20):
    f = open(file_name, "r")
    freader = csv.reader(f, delimiter = ',')

    next(freader) #skip header
    data = []
    labels = []
    pix = []
    for row in freader:
        float_row = [float(i) for i in row]
        data.append(float_row)
        labels.append(float_row[0:1])
        pix.append(float_row[1:])
        limit -= 1
        if limit <= 0:
            break
    return data, labels, pix

def normalize_data(m):
    return m / m.max()

def load_train_data():
    limit = int(utils.find_argv("datalines", "20"))
    data, labels, pix = read_data(train_file_name, limit)
    #print("size of data: {}".format(sys.getsizeof(data)))

    sample_count = len(labels)
    
    train_count = int(0.8 * sample_count)

    train_pix = pix[:train_count]
    train_labels = labels[:train_count]

    test_pix = pix[train_count:]
    test_labels = labels[train_count:]
    return train_pix, train_labels, test_pix, test_labels

