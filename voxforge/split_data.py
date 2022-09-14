import numpy as np
import pandas as pd
import argparse, pathlib


ONE_IN_X = 5


def split_by_bin_packing(src_path, trgt_langs):
    path = pathlib.Path(src_path).parent

    df = pd.read_csv(src_path, index_col=0)
    langs = df.columns[-1]

    train_list = []
    test_list = []

    for name, data in df.groupby(langs):

        if not name in trgt_langs:
            continue

        #Retrieve speakers with the respective amount of recordings
        speakers : pd.Series
        speakers = data.groupby('speaker').size()
        
        #Offline bin-packing => sorting desc, then first-fit
        speakers = speakers.sort_values(ascending=False)

        bin_size = speakers.sum() // ONE_IN_X
        bins = [
            [bin_size, []]
        ]

        #pack bins first fit
        for name, size in speakers.items():
            fits = None
            for b in bins:
                if b[0] > size:
                    fits = b
                    break
            if fits is None:
                fits = [bin_size, []]
                bins.append(fits)
            fits[0] -= size
            fits[1].append(name)

        #Get most filled bin as test set, rest is training set
        test_idx = None
        test_size = bin_size
        for i in range(len(bins)):
            if np.abs( bins[i][0] ) < test_size:
                test_size = bins[i][0]
                test_idx = i

        for i in range(len(bins)):
            entries = [ data[ data['speaker'] == name ] for name in bins[i][1] ]
            if i == test_idx:
                test_list.append( pd.concat(entries, ignore_index=True) )
            else:
                train_list.append( pd.concat(entries, ignore_index=True) )

    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)

    train_df.rename(columns={ train_df.columns[-1]: ' '.join(trgt_langs) })
    test_df.rename(columns={ test_df.columns[-1]: ' '.join(trgt_langs) })

    train_df.to_csv(path/f"train_{','.join(trgt_langs)}_data.csv")
    test_df.to_csv(path/f"test_{','.join(trgt_langs)}_data.csv")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split a dataset into training and test data whilst selecting specific languages")
    parser.add_argument('-l', '--lang', action="append",
        help="The languages that should be included in the resulting dataset, given once per language")
    parser.add_argument('-m', '--mag', default=5, type=int,
        help="Specifies the magnitude/factor by which the training set is larger than the test set")
    parser.add_argument('path')
    args = parser.parse_args()
    ONE_IN_X = args.mag
    split_by_bin_packing(args.path, args.lang)
