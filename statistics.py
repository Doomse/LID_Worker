import torch, numpy as np
from torch.utils import data
import torch_models, torch_datasets


MODEL = 'models/de,en,es,fr,it,ru__3__0.001____2022_07_19__16_26_57'
DATASET = '/export/data1/data/dhoefer/voxforge/set_001/test_data.csv'
BATCH_SIZE = 256
KERNEL_SIZE = 3

GPU_ID = 4

if torch.cuda.is_available():
    DEVICE = torch.device('cuda', GPU_ID)
    print('use cuda')
else:
    DEVICE = torch.device('cpu')


def init_model():
    dl = data.DataLoader(torch_datasets.VoxforgeDataset(DATASET), batch_size=BATCH_SIZE, shuffle=True)
    model = torch_models.Orig1d(langs=len(dl.dataset.langs), kernel_size=KERNEL_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(MODEL))
    model.eval()
    return model, dl


def confusion_matrix(model, dl):
    langs = dl.dataset.langs
    totals = np.zeros( len(langs) )
    hits = np.zeros( ( len(langs), len(langs), ) ) #1st dim trues, 2nd dim guesses

    with torch.no_grad():
        for x, y in dl:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)

            for guess, true in zip( pred.argmax(1), y ):
                totals[true] += 1
                hits[true, guess] += 1
    
    return hits / totals


if __name__ == '__main__':
    model, dl = init_model()
    langs = dl.dataset.langs
    conf_matrix = confusion_matrix(model, dl)
    print(f"Trues \ Guesses   {', '.join(langs)}")
    with np.printoptions(precision=3, suppress=True):
        print(conf_matrix)

