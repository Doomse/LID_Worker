import numpy as np
import pandas as pd
import pathlib


ONE_IN_X = 5


def split_by_bin_packing(src_path):
    path = pathlib.Path(src_path).parent

    df = pd.read_csv(src_path, index_col=0)
    langs = df.columns[-1]

    train_list = []
    test_list = []

    for _, data in df.groupby(langs):

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

    train_df.to_csv(path/'train_data.csv')
    test_df.to_csv(path/'test_data.csv')



if __name__ == '__main__':
    split_by_bin_packing('/home/dhoefer/full_data.csv')
