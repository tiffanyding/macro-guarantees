"""
Download softmax scores and labels for a dataset, combine cal+test,
and save to data/{dataset}/. 

Usage:
    python download_data.py [plantnet|plantnet-trunc|inaturalist-trunc]
"""
import argparse
import os
import subprocess
import tempfile
import zipfile

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    'plantnet': {
        'gdrive_id': '1k_PPQV3VJT44hz02CcnbqPstjQo70vGr',
        'prefix':    'best-plantnet-model',
    },
    'plantnet-trunc': {
        'gdrive_id': '1a0SF6xbMDwxmAde2VgBoP4q_qYX2hslm',
        'prefix':    'best-plantnet-trunc-model',
    },
    'inaturalist-trunc': {
        'gdrive_id': '1patL2K450vwiI4DGugWlCPXh-j6y6uCF',
        'prefix':    'best-inaturalist-trunc-model',
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', nargs='?', default='plantnet-trunc',
                        choices=list(DATASETS.keys()))
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    out_dir = os.path.join(_THIS_DIR, 'data', args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, f'{args.dataset}.zip')

        print(f'Downloading {args.dataset} scores from Google Drive...')
        subprocess.run(['gdown', cfg['gdrive_id'], '-O', zip_path], check=True)

        print('Extracting and combining cal + test arrays...')
        p = cfg['prefix']
        with zipfile.ZipFile(zip_path) as z:
            cal_softmax  = np.load(z.open(f'{p}_cal_softmax.npy'))
            cal_labels   = np.load(z.open(f'{p}_cal_labels.npy'))
            test_softmax = np.load(z.open(f'{p}_test_softmax.npy'))
            test_labels  = np.load(z.open(f'{p}_test_labels.npy'))
            train_labels = np.load(z.open(f'{args.dataset}_train_labels.npy'))

    softmax = np.concatenate([cal_softmax, test_softmax], axis=0)
    labels  = np.concatenate([cal_labels,  test_labels],  axis=0)

    print(f'Combined: softmax {softmax.shape}, labels {labels.shape}')

    np.save(os.path.join(out_dir, 'softmax_scores.npy'), softmax)
    np.save(os.path.join(out_dir, 'labels.npy'), labels)
    np.save(os.path.join(out_dir, f'train_labels.npy'), train_labels)


    print(f'Saved to {out_dir}/')
    print(f'  softmax_scores.npy  {softmax.shape}  {softmax.dtype}')
    print(f'  labels.npy          {labels.shape}  {labels.dtype}')
    print(f'  train_labels.npy    {train_labels.shape}  {train_labels.dtype}')

if __name__ == '__main__':
    main()
