import numpy as np
import pandas as pd
import argparse, pathlib



def clamp_for_speaker_amount(src_path, size):
    path = pathlib.Path(src_path)

    df = pd.read_csv(src_path, index_col=0)
    langs = df.columns[-1]

    df_list = []

    for name, data in df.groupby(langs):

        assert len(data) >= size

        speakers = data.groupby('speaker')

        running = True
        offset = 0
        max_len = size // len(speakers)
        cur_size = 0

        while running:
            for uname, spk in speakers:

                if len(spk) < offset + max_len:

                    if len(spk) < offset:
                        continue

                    cur_size += len(spk) - offset

                    if cur_size >= size:
                        df_list.append( spk[ offset : len(spk) - (cur_size - size) ] )
                        running = False
                        break

                    df_list.append( spk[ offset : len(spk) ] )

                else:
                    
                    cur_size += max_len

                    if cur_size  >= size:
                        df_list.append( spk[ offset : offset + max_len - (cur_size - size) ] )
                        running = False
                        break

                    df_list.append( spk[ offset : offset + max_len ] )

            offset += max_len
            max_len = (size - cur_size) // len(speakers)
            max_len = max(max_len, 1)

    new_df = pd.concat(df_list, ignore_index=True)

    new_df.to_csv(path.parent/f"{path.stem}_{size}{path.suffix}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clamp a dataset at a certain length per language")
    parser.add_argument('-s', '--size', type=int, help="The size per language")
    parser.add_argument('path')
    args = parser.parse_args()
    clamp_for_speaker_amount(args.path, args.size)
    

        

