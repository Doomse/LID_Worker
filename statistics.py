import torch, numpy as np
from torch.utils import data
import torch_models, torch_datasets
import argparse, pathlib



def init_model(model_path, dataset_path, batch_size):
    path = pathlib.Path(model_path)
    langs, kernel_size, lr, dropout, *_ = path.name.split('__')
    langs = langs.split(',')
    kernel_size = int(kernel_size)
    lr = float(lr)

    dl = data.DataLoader(torch_datasets.VoxforgeDataset(dataset_path), batch_size=batch_size, shuffle=True)
    assert dl.dataset.langs == langs
    if dropout:
        dropout = float(dropout)
        model = torch_models.Orig1d(langs=len(dl.dataset.langs), kernel_size=kernel_size, dropout=dropout).to(DEVICE)
    else:
        model = torch_models.Orig1d(langs=len(dl.dataset.langs), kernel_size=kernel_size).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model, dl


def confusion_matrix(model, dl, length, offset):
    langs = dl.dataset.langs
    totals = np.zeros( ( len(langs), 1, ) )
    hits = np.zeros( ( len(langs), len(langs), ) ) #1st dim trues, 2nd dim guesses

    with torch.no_grad():
        for x, y in dl:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            for i in range(0, x.shape[-1] - length + 1, offset):

                x0 = x[..., i:i+length]

                pred = model(x0)

                for guess, true in zip( pred.argmax(1), y ):
                    totals[true, 0] += 1
                    hits[true, guess] += 1
    
    return hits / totals


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculates the confusion matrix for a model and dataset")
    parser.add_argument('--gpu', type=int, help="GPU id to use (if available)")
    parser.add_argument('-m', '--model', help="File holding the model parameters")
    parser.add_argument('-d', '--dataset', help="Dataset to test with")
    parser.add_argument('-b', '--batchsize', type=int, default=256, help="Batch size to run, default 256")
    parser.add_argument('-l', '--length', type=int, default=80000, help="Length of subsegments")
    parser.add_argument('-o', '--offset', type=int, default=80000, help="Offset of subsegments")
    args = parser.parse_args()

    if torch.cuda.is_available() and args.gpu:
        DEVICE = torch.device('cuda', args.gpu)
        print('use cuda')
    else:
        DEVICE = torch.device('cpu')

    model, dl = init_model(args.model, args.dataset, args.batchsize)
    langs = dl.dataset.langs
    conf_matrix = confusion_matrix(model, dl, args.length, args.offset)
    print(f"Trues \ Guesses   {', '.join(langs)}")
    with np.printoptions(precision=3, suppress=True):
        print(conf_matrix)

