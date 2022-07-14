import sys
import os  
from tqdm import tqdm
import kaldiark
import numpy as np 

def read_scp_file(scp_path, data_dir):
    with open(scp_path, 'r') as fp:
        data = {}
        for x in fp:
            key, path = x.split(' ')
            name, bits = path.split('/')[-1].split(':')
            data[key] = (os.path.join(data_dir, name), int(bits))
    return data 

def read_mean_var(path):
    with open(path, 'r') as fp:
        sum_ = np.fromstring(fp.readline().strip()[1:-1], dtype=float, sep=',')
        sum_sqr =  np.fromstring(fp.readline().strip()[1:-1], dtype=float, sep=',')
        n_frame = int(fp.readline().strip())
        mean = sum_/n_frame
        std = np.power((sum_sqr/n_frame)-(np.power(mean,2)), 1/2)
        return mean, std

def main(
    data_dir,
    out_dir,
):
    key_label_path = os.path.join(data_dir, 'train_960.hubert8.bas.scp')
    os.makedirs(out_dir, exist_ok=True)
    label_dict = read_scp_file(key_label_path, data_dir)
    
    for key, values in tqdm(label_dict.items()):
        data_path, bits = values[0], values[1]
        with open(data_path, 'r') as fp:
            fp.seek(bits)
            label = np.array(list(map(int, fp.readline().strip().split(' '))))
            assert not ((label >= 512).any() or (label < 0).any())
            save_path = os.path.join(out_dir, key+'.npy')
            np.save(save_path, label)

if __name__ == "__main__":
    data_dir, out_dir = sys.argv[1], sys.argv[2]
    main(data_dir, out_dir)
