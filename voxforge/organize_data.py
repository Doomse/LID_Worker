import torch, torchaudio
import math, random
import pandas
import pathlib, subprocess


BASE_DIR = pathlib.Path(__file__).resolve().parent

ONE_IN_X = 5 #One in x files will be assigned to the test set

TARGET_SAMPLE_RATE = 8000
TARGET_LENGTH = 10

GPU_ID = 4

if torch.cuda.is_available():
    DEVICE = torch.device('cuda', GPU_ID)
else:
    DEVICE = torch.device('cpu')


def extract_tar_archives(dry_run=False):
    filenames = {}
    langs_dir = BASE_DIR.glob('*')
    for lang_dir in langs_dir:

        if lang_dir.is_file(): #Skip files
            continue

        filenames[lang_dir] = []
        for file in lang_dir.rglob('*.tgz'):
            if not dry_run:
                subprocess.run(['tar', '--extract', '--file', file.name], cwd=file.parent)
            filenames[lang_dir].append(file.parent/file.stem)

    return filenames

def create_training_test_data(filenames: dict[pathlib.Path,list[pathlib.Path]]):
    train_entries = []
    test_entries = []
    langs = []
    for lang_dir, audio_dirlist in filenames.items():
        split_counter = 1
        for audio_dir in audio_dirlist:
            target_path = lang_dir/audio_dir.parent.name
            target_name = audio_dir.name

            audio_dir /= 'wav'

            total_audio_frames = 0
            total_audio = []
            prev_sample_rate = 0
            resampler = None
            for file in sorted(audio_dir.iterdir(), key=lambda x: x.name):
                audio, sample_rate = torchaudio.load(file)
                audio = audio.to(DEVICE)
                audio_frames = audio.shape[1]

                #Finish file
                if total_audio_frames + audio_frames >= TARGET_LENGTH*sample_rate:

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
                    if sample_rate != prev_sample_rate:
                        resampler = torchaudio.transforms.Resample(sample_rate, TARGET_SAMPLE_RATE).to(DEVICE)
                        prev_sample_rate = sample_rate
                    waveform = resampler(waveform)
                    assert waveform.shape[1] == TARGET_LENGTH*TARGET_SAMPLE_RATE

                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.touch()
                    torchaudio.save(target, waveform, TARGET_SAMPLE_RATE)

                    if math.remainder(split_counter, 5):
                        train_entries.append( ( str(target), lang_dir.name, ) )
                        split_counter += 1
                    else:
                        test_entries.append( ( str(target), lang_dir.name, ) )
                        split_counter = 1


                    total_audio_frames = audio_frames - suf_frames
                    total_audio = [ audio[:,suf_frames:] ]
                    continue
                    
                total_audio_frames += audio_frames
                total_audio.append(audio)

        langs.append(lang_dir.name)

    columns = ['path', ' '.join(langs)]
    train_frame = pandas.DataFrame(data=train_entries, columns=columns)
    train_frame.to_csv(BASE_DIR/'train_data.csv')
    test_frame = pandas.DataFrame(data=test_entries, columns=columns)
    test_frame.to_csv(BASE_DIR/'test_data.csv')

        



if __name__ == '__main__':
    filenames = extract_tar_archives()
    create_training_test_data(filenames)