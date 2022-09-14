import glob
import pandas as pd

for PART in ['dev', 'test', 'train']:

    files = glob.glob(f'C:/Users/User/Desktop/paraph/**/distances/{PART}_*.csv',
                      recursive=True)

    # Join (concatenate) dataframes with corresponding features
    df = pd.concat(map(pd.read_csv, files), axis=1)
    # Drop Id column
    df.drop(columns='id', inplace=True)

    # Save joined features to csv
    df.to_csv(f'data/features/{PART}.csv', index=False)
    print('ready')
