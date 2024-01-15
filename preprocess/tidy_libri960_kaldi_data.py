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
    # Data path
    fbank_data_dir = os.path.join(data_dir, 'fbank')
    kmeans_dir_10ms = os.path.join(data_dir, 'stage2-cluster-10ms')
    kmeans_dir_20ms = os.path.join(data_dir, 'stage2-cluster-20ms')
    
    key_data_path = os.path.join(fbank_data_dir, 'train-960.scp')
    mean_var_path = os.path.join(fbank_data_dir, 'train-960.mean-var')
    data_save_dir = os.path.join(out_dir, 'feature')
    os.makedirs(data_save_dir, exist_ok=True)
    
    mean, std = read_mean_var(mean_var_path)
    mean_std = np.concatenate((mean.reshape(1,-1), std.reshape(1,-1)), axis=0)
    mean_std_npy_save_path = os.path.join(out_dir, 'mean-std.npy')
    np.save(mean_std_npy_save_path, mean_std)
    data_dict = read_scp_file(key_data_path, fbank_data_dir)
    
    for fp in ['10ms', '20ms']:
        kmeans_dir = eval(f'kmeans_dir_{fp}')
        key_label_path = os.path.join(kmeans_dir, 'train_960.hubert8.bas.scp')
        label_save_dir = os.path.join(out_dir, f'cluster_{fp}')
        out_csv = os.path.join(out_dir, f'libri960-stg2-{fp}.csv')
        os.makedirs(label_save_dir, exist_ok=True)
        out_fp = open(out_csv, 'w')
    
        label_dict = read_scp_file(key_label_path, kmeans_dir)
        recorder = {}
   
        for key, values in tqdm(label_dict.items()):
            data_path, bits = values[0], values[1]
            with open(data_path, 'r') as fp:
                fp.seek(bits)
                label = np.array(list(map(int, fp.readline().strip().split(' '))))
                assert not ((label >= 512).any() or (label < 0).any()), f'got {label}'
                length = label.shape[0]
                save_path = os.path.join(label_save_dir, key+'.npy')
                recorder[key] = [save_path, length]
                np.save(save_path, label)

        for key, values in tqdm(data_dict.items()):
            data_path, bits = values[0], values[1]
            with open(data_path, 'rb') as fp:
                fp.seek(bits)
                feat = kaldiark.parse_feat_matrix(fp)
                feat = (feat-mean)/std
            length = feat.shape[0]
            save_path = os.path.join(data_save_dir, key+'.npy')
            recorder[key].append(save_path)
            np.save(save_path, feat)
    
        out_fp.write('file_path,label_path,length\n')
        for label_path, length, data_path in recorder.values():
            out_fp.write(f'{data_path},{label_path},{length}\n')

if __name__ == "__main__":
    data_dir, out_dir = sys.argv[1], sys.argv[2]
    main(data_dir, out_dir)
