import logging, pathlib, torch, torchaudio, numpy as np
import torch_models

lid_logger = logging.getLogger('lid')

BUFFER_LENGTH = 80000


# Currently doesn't support batched input
class ModelRunner:

    def __init__(self, p: str, sr: int):

        path = pathlib.Path(p)
        langs, kernel_size, lr, *_ = path.name.split('__')
        self.langs = langs.split(',')
        kernel_size = int(kernel_size)
        lr = float(lr)
        #TODO handle dropout

        self.model = torch_models.Orig1d(langs=len(self.langs), kernel_size=kernel_size)
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.model.eval()

        self.buffer = np.empty(0, dtype=np.int16)
        self.buf_len = BUFFER_LENGTH

        self.resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=8000)

        self.softmax = torch.nn.Softmax(dim=-1)


    def run_model(self, a: np.ndarray):

        self.buffer = np.concatenate( (self.buffer, a, ) )

        lid_logger.debug(f"{self.buffer.shape=}")

        if len(self.buffer) >= self.buf_len:

            audio_orig = self.buffer[:self.buf_len]
            self.buffer = self.buffer[self.buf_len:]

            lid_logger.debug(f"{audio_orig.shape=}")

            # Normalize to [-1.0, 1.0] range
            audio = torch.from_numpy(audio_orig)
            audio = audio / torch.iinfo(audio.dtype).max
            audio = audio[None, ...] # Add channel dimension

            # Run resample + model + softmax
            lid_logger.debug(f"{audio.shape=}")
            input = self.resampler(audio)
            lid_logger.debug(f"{input.shape=}")
            output = self.model( input[None, ...] )[0] # Add and remove batch dimension to make batch-norm happy
            lid_logger.debug(f"{output=}")
            probs = self.softmax(output)

            lid_logger.debug(f"{','.join(self.langs)}: {probs}")

            return audio_orig, self.langs[ torch.argmax(probs) ]

    def finish(self):

        audio_orig = self.buffer

        # Normalize to [-1.0, 1.0] range
        audio = torch.from_numpy(audio_orig)
        audio = audio / torch.iinfo(audio.dtype).max
        audio = torch.mean(audio, dim=0, keepdim=True) # Downmix to mono according to https://github.com/pytorch/audio/issues/363 

        # Run resample + model + softmax
        input = self.resampler(audio)
        lid_logger.debug(f"{input.shape=}")
        output = self.model( input[None, ...] )[0] # Add and remove batch dimension to make batch-norm happy
        lid_logger.debug(f"{output=}")
        probs = self.softmax(output)

        lid_logger.debug(f"{','.join(self.langs)}: {probs}")

        return audio_orig, self.langs[ torch.argmax(probs) ]



