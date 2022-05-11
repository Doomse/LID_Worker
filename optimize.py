import torch
from torch.utils import data
import torch_models, torch_datasets
import datetime, pathlib

EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
LANGS = ['de', 'en']
WEIGHTS = [0.7, 0.3]

BASE_DIR = pathlib.Path(__file__).resolve().parent

GPU_ID = 4

if torch.cuda.is_available():
    DEVICE = torch.device('cuda', GPU_ID)
else:
    DEVICE = torch.device('cpu')


def optimize(epochs, batch_size, learning_rate, langs, train_ds_loc, test_ds_loc):
    logfile = BASE_DIR/'logs'
    logfile.mkdir(exist_ok=True)
    logfile /= datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    logfile.touch()

    modelfile = BASE_DIR/'models'
    modelfile.mkdir(exist_ok=True)
    modelfile /= f"{','.join(langs)}____{datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}"
    modelfile.touch()

    model = torch_models.Orig1d(langs=len(LANGS)).to(DEVICE)
    train_dl = data.DataLoader(torch_datasets.VoxforgeDataset(train_ds_loc), batch_size=batch_size, shuffle=True)
    test_dl = data.DataLoader(torch_datasets.VoxforgeDataset(test_ds_loc), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.Tensor(WEIGHTS)).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)

    with logfile.open('w') as log:
        for ep in range(epochs):
            # Training loop
            model.train()
            for x, y in train_dl:
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                pred = model(x)
                loss = loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            #Test loop
            model.eval()
            ds_size, num_batches, loss_sum, correct = 0, 0, 0, 0
            with torch.no_grad():
                for x, y in test_dl:
                    x = x.to(DEVICE)
                    y = y.to(DEVICE)

                    pred = model(x)
                    loss_sum += loss_fn(pred, y).item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()

                    ds_size += batch_size
                    num_batches += 1
            loss_avg = loss_sum/num_batches
            correct_rel = correct/ds_size
            log.write(f"Test Error (Epoch {ep}): \n Accuracy: {(100*correct_rel):>0.1f}%, Avg loss: {loss_avg:>8f} \n")
    
    torch.save(model.state_dict(), modelfile)





if __name__ == '__main__':
    optimize(EPOCHS, BATCH_SIZE, LEARNING_RATE, LANGS, '/home/domse/bch/voxforge/train_data.csv', '/home/domse/bch/voxforge/test_data.csv')