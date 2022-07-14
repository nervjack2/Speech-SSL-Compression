import os 
import sys 
import numpy as np
from tqdm import tqdm
from os.path import join

kmeans_dir, feat_dir, out_csv = sys.argv[1], sys.argv[2], sys.argv[3]
paths = os.listdir(kmeans_dir)
out_fp = open(out_csv, 'w')
out_fp.write('file_path,label_path,length\n')

for path in tqdm(paths):
    label = np.load(join(kmeans_dir,path))
    length = label.shape[0]
    out_fp.write(f'{join(feat_dir,path)},{join(kmeans_dir,path)},{length}\n')