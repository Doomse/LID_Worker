import torch, torchaudio
import random
import pandas
import pathlib, subprocess
import datetime, traceback
import argparse



def extract_tar_archives(dry_run=False):
    logfile = BASE_DIR/'logs'
    logfile.mkdir(exist_ok=True)
    logfile /= datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    logfile.touch()

    with logfile.open('w') as log:
        filenames = {}
        langs_dir = BASE_DIR.glob('*')
        for lang_dir in langs_dir:

            if lang_dir.is_file(): # Skip files
                continue
            if lang_dir.name == 'logs': # Skip logs
                continue
            if lang_dir.name.startswith('set'): # Skip existing sets
                continue

            filenames[lang_dir] = []
            for file in lang_dir.rglob('*.tgz'):
                if not dry_run:
                    subprocess.run(['tar', '--extract', '--file', file.name], cwd=file.parent, stdout=log, stderr=subprocess.STDOUT)
                filenames[lang_dir].append(file.parent/file.stem)

    return filenames

def create_full_dataset(filenames: dict[pathlib.Path,list[pathlib.Path]], shuffle=False):
    logfile = BASE_DIR/'logs'
    logfile.mkdir(exist_ok=True)
    logfile /= datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    logfile.touch()

    with logfile.open('w') as log:
        try:
            entries = []
            langs = []

            for i in range(1000):
                set_dir = BASE_DIR/f"set_{i:03d}"
                if not set_dir.exists():
                    break
            set_dir.mkdir(parents=True, exist_ok=True)

            for lang_dir, audio_dirlist in filenames.items():

                for audio_dir in sorted(audio_dirlist, key=lambda x: x.name):
                    try:
                        target_path = set_dir/lang_dir.name/audio_dir.parent.name
                        target_name = audio_dir.name

                        speaker, rec_date, *_ = target_name.split('-')

                        #Locate audio dir
                        new_audio_dir = None
                        for dirname in audio_dir.iterdir():
                            if not dirname.is_dir(): # Skip files
                                continue
                            if dirname.name == 'etc': # Skip etc
                                continue
                            if dirname.name == 'wav': # Prefer wav
                                new_audio_dir = dirname
                                continue
                            if new_audio_dir is None: # Use first dir found
                                new_audio_dir = dirname
                                continue

                        if new_audio_dir is None:
                            continue
                        if new_audio_dir.name != 'wav':
                            print("Using non-wav format in ", new_audio_dir)
                            log.write(f"Using non-wav format in {new_audio_dir}\n")
                        audio_dir = new_audio_dir

                        total_audio_frames = 0
                        total_audio = []
                        prev_sample_rate = 0
                        resampler = None

                        dir_list = sorted(audio_dir.iterdir(), key=lambda x: x.name)
                        if shuffle:
                            random.shuffle(dir_list)
                        
                        #Concat and splice audio files
                        for file in dir_list:
                            audio, sample_rate = torchaudio.load(file)
                            audio = audio.to(DEVICE)
                            audio_frames = audio.shape[1]

                            #Finish file
                            while total_audio_frames + audio_frames >= TARGET_LENGTH*sample_rate:

                                suf_frames = TARGET_LENGTH*sample_rate - total_audio_frames
                                total_audio.append(audio[:,:suf_frames])

                                for i in range(0, 1000, 1):
                                    target = target_path/f'{target_name}__{i}.wav'
                                    if not target.exists():
                                        break
                                
                                waveform = torch.hstack(total_audio)
                                assert waveform.shape[1] == TARGET_LENGTH*sample_rate
                                waveform = waveform.to(DEVICE) #TODO Maybe not necessary
                                waveform = torch.mean(waveform, dim=0, keepdim=True) # Downmix to mono according to https://github.com/pytorch/audio/issues/363
                                
                                if TARGET_SAMPLE_RATE is not None:
                                    if sample_rate != prev_sample_rate:
                                        resampler = torchaudio.transforms.Resample(sample_rate, TARGET_SAMPLE_RATE).to(DEVICE)
                                        prev_sample_rate = sample_rate
                                    waveform = resampler(waveform)
                                    assert waveform.shape[1] == TARGET_LENGTH*TARGET_SAMPLE_RATE

                                target.parent.mkdir(parents=True, exist_ok=True)
                                target.touch()
                                waveform = waveform.cpu()
                                torchaudio.save(target, waveform, TARGET_SAMPLE_RATE)


                                entries.append( ( str(target), speaker, lang_dir.name, ) )


                                audio_frames = audio_frames - suf_frames
                                audio = audio[:,suf_frames:]
                                total_audio_frames = 0
                                total_audio = []
                                
                            total_audio_frames += audio_frames
                            total_audio.append(audio)
                    except Exception:
                        traceback.print_exc()
                        traceback.print_exc(file=log)

                langs.append(lang_dir.name)

        finally:
            columns = ['path', 'speaker', ' '.join(langs)]
            train_frame = pandas.DataFrame(data=entries, columns=columns)
            train_frame.to_csv(set_dir/'full_data.csv')

        



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extracts the data from a given voxforge dataset and collects it in a csv archive")
    parser.add_argument('--gpu', type=int, help="GPU id to use (if available)")
    parser.add_argument('-r', '--samplerate', type=int, help="Target sample rate, leave blank for no resampling")
    parser.add_argument('-l', '--length', type=int, default=10, help="Target length of segments in seconds. Defaults to 10")
    args = parser.parse_args()

    BASE_DIR = pathlib.Path(__file__).resolve().parent

    TARGET_SAMPLE_RATE = args.samplerate
    TARGET_LENGTH = args.length

    if torch.cuda.is_available() and args.gpu:
        DEVICE = torch.device('cuda', args.gpu)
        print('use cuda')
    else:
        DEVICE = torch.device('cpu')

    filenames = extract_tar_archives()
    create_full_dataset(filenames)
