#!/usr/bin/env python3

import sys
import os  
import kaldiark
import numpy as np 

def usage():
    print(
        'usage: tidy_libri360_kaldi_data.py fbank.scp bas.scp fbank.mean-var feature/ cluster/ mean-std.npy out.csv'
    )


def read_scp_file(scp_path):
    data = {}
    with open(scp_path, 'r') as fp:
        for x in fp:
            key, path_bits = x.split(' ')
            path, bits = path_bits.split(':')
            data[key] = (path, int(bits))
    return data 


def read_mean_var(path):
    with open(path, 'r') as fp:
        sum_ = np.fromstring(fp.readline().strip()[1:-1], dtype=float, sep=',')
        sum_sqr =  np.fromstring(fp.readline().strip()[1:-1], dtype=float, sep=',')
        n_frame = int(fp.readline().strip())
        mean = sum_ / n_frame
        std = np.power(sum_sqr / n_frame - np.power(mean, 2), 1/2)
        return mean, std


def main(
    key_data_path, key_label_path, mean_var_path,
    data_save_dir, label_save_dir, mean_std_npy_save_path,
    out_csv
):
    out_fp = open(out_csv, 'w')
    mean, std = read_mean_var(mean_var_path)
    mean_std = np.concatenate((mean.reshape(1,-1), std.reshape(1,-1)), axis=0)
    np.save(mean_std_npy_save_path, mean_std)
    data_dict = read_scp_file(key_data_path)
    label_dict = read_scp_file(key_label_path)
    recorder = {}
    
    for key, values in data_dict.items():
        data_path, bits = values[0], values[1]
        with open(data_path, 'rb') as fp:
            fp.seek(bits)
            feat = kaldiark.parse_feat_matrix(fp)
            feat = (feat-mean)/std
            length = feat.shape[0]
            save_path = os.path.join(data_save_dir, key+'.npy')
            recorder[key] = [save_path, length]
            np.save(save_path, feat)
    
    for key, values in label_dict.items():
        data_path, bits = values[0], values[1]
        with open(data_path, 'r') as fp:
            fp.seek(bits)
            label = np.array(list(map(int, fp.readline().strip().split(' '))))
            assert not ((label >= 512).any() or (label < 0).any())
            length = label.shape[0]
            save_path = os.path.join(label_save_dir, key+'.npy')
            recorder[key].append(save_path)
            assert recorder[key][1] == length
            np.save(save_path, label)

    out_fp.write('file_path,label_path,length\n')
    for data_path, length, label_path in recorder.values():
        out_fp.write(f'{data_path},{label_path},{length}\n')

if __name__ == "__main__":
    if len(sys.argv) == 1:
        usage()
        exit()

    key_data_path, key_label_path, mean_var_path = sys.argv[1], sys.argv[2], sys.argv[3]
    data_save_dir, label_save_dir = sys.argv[4], sys.argv[5]
    mean_std_npy_save_path, out_csv = sys.argv[6], sys.argv[7]

    main(key_data_path, key_label_path, mean_var_path, data_save_dir,
        label_save_dir, mean_std_npy_save_path, out_csv)

