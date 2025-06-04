import os, logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def generate_metadata_chexpert(data_path, test_pct=0.15, val_pct=0.1):
    logging.info("Generating metadata for CheXpert No Finding prediction...")
    chexpert_dir = Path(data_path)
    assert (chexpert_dir/'train.csv').is_file()
    assert (chexpert_dir/'train/patient48822/study1/view1_frontal.jpg').is_file()
    assert (chexpert_dir/'valid/patient64636/study1/view1_frontal.jpg').is_file()
    assert (chexpert_dir/'CHEXPERT DEMO.xlsx').is_file()

    df = pd.concat([pd.read_csv(chexpert_dir/'train.csv'), pd.read_csv(chexpert_dir/'valid.csv')], ignore_index=True)

    df['filename'] = df['Path'].astype(str).apply(lambda x: os.path.join(chexpert_dir, x[x.index('/')+1:]))
    df['subject_id'] = df['Path'].apply(lambda x: int(Path(x).parent.parent.name[7:])).astype(str)
    df = df[df.Sex.isin(['Male', 'Female'])]
    details = pd.read_excel(chexpert_dir/'CHEXPERT DEMO.xlsx', engine='openpyxl')[['PATIENT', 'PRIMARY_RACE']]
    details['subject_id'] = details['PATIENT'].apply(lambda x: x[7:]).astype(int).astype(str)

    df = pd.merge(df, details, on='subject_id', how='inner').reset_index(drop=True)

    def cat_race(r):
        if isinstance(r, str):
            if r.startswith('White'):
                return 0
            elif r.startswith('Black'):
                return 1
        return 2

    df['ethnicity'] = df['PRIMARY_RACE'].apply(cat_race)
    attr_mapping = {'Male_0': 0, 'Female_0': 1, 'Male_1': 2, 'Female_1': 3, 'Male_2': 4, 'Female_2': 5}
    df['a'] = (df['Sex'] + '_' + df['ethnicity'].astype(str)).map(attr_mapping)
    df['y'] = df['No Finding'].fillna(0.0).astype(int)

    train_val_idx, test_idx = train_test_split(df.index, test_size=test_pct, random_state=42, stratify=df['a'])
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_pct/(1-test_pct), random_state=42, stratify=df.loc[train_val_idx, 'a'])

    df['split'] = 0
    df.loc[val_idx, 'split'] = 1
    df.loc[test_idx, 'split'] = 2

    (chexpert_dir/'subpop_bench_meta').mkdir(exist_ok=True)
    df.to_csv(os.path.join(chexpert_dir, 'subpop_bench_meta', "metadata_no_finding.csv"), index=False)

def generate_metadata_metashift(data_path, test_pct=0.25, val_pct=0.1):
    logging.info("Generating metadata for MetaShift...")
    dirs = {
        'train/cat/cat(indoor)': [1, 1],
        'train/dog/dog(outdoor)': [0, 0],
        'test/cat/cat(outdoor)': [1, 0],
        'test/dog/dog(indoor)': [0, 1]
    }
    ms_dir = os.path.join(data_path, "metashift")

    all_data = []
    for dir in dirs:
        folder_path = os.path.join(ms_dir, 'MetaShift-Cat-Dog-indoor-outdoor', dir)
        y = dirs[dir][0]
        g = dirs[dir][1]
        for img_path in Path(folder_path).glob('*.jpg'):
            all_data.append({
                'filename': img_path,
                'y': y,
                'a': g
            })
    df = pd.DataFrame(all_data)

    rng = np.random.RandomState(42)

    test_idxs = rng.choice(np.arange(len(df)), size=int(len(df) * test_pct), replace=False)
    val_idxs = rng.choice(np.setdiff1d(np.arange(len(df)), test_idxs), size=int(len(df) * val_pct), replace=False)

    split_array = np.zeros((len(df), 1))
    split_array[val_idxs] = 1
    split_array[test_idxs] = 2

    df['split'] = split_array.astype(int)
    df.to_csv(os.path.join(ms_dir, "metadata_metashift.csv"), index=False)

if __name__ == "__main__":
    # generate_metadata_chexpert('/home/sci/hdai/Projects/Datasets/CheXpert-v1.0-small')

    generate_metadata_metashift('/home/sci/hdai/Projects/Datasets')