import torch
from torch.utils import data
from torch.cuda import amp
import torch_models, torch_datasets
import datetime, pathlib

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

KERNEL_SIZE = 3

BASE_DIR = pathlib.Path(__file__).resolve().parent

GPU_ID = 4

if torch.cuda.is_available():
    DEVICE = torch.device('cuda', GPU_ID)
    print('use cuda')
else:
    DEVICE = torch.device('cpu')


def optimize(epochs, batch_size, learning_rate, kernel_size, train_ds_loc, test_ds_loc):
    logfile = BASE_DIR/'logs'
    logfile.mkdir(exist_ok=True)
    logfile /= datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    logfile.touch()

    train_dl = data.DataLoader(torch_datasets.VoxforgeDataset(train_ds_loc), batch_size=batch_size, shuffle=True)
    test_dl = data.DataLoader(torch_datasets.VoxforgeDataset(test_ds_loc), batch_size=batch_size, shuffle=True)
    model = torch_models.Orig1d(langs=len(train_dl.dataset.langs), kernel_size=kernel_size).to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.Tensor(train_dl.dataset.weights)).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)
    scaler = amp.GradScaler()

    print(f"Langs: {train_dl.dataset.langs}\nTrain weights: {train_dl.dataset.weights}\nTest weights: {test_dl.dataset.weights}")

    modelfile = BASE_DIR/'models'
    modelfile.mkdir(exist_ok=True)
    modelfile /= f"{','.join(train_dl.dataset.langs)}__{KERNEL_SIZE}__{LEARNING_RATE}____{datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}"
    modelfile.touch()

    with logfile.open('w') as log:

        log.write(f"Langs: {','.join(train_dl.dataset.langs)}\nKernel size: {KERNEL_SIZE}\nLearning rate: {LEARNING_RATE}\n\n\n")

        for ep in range(epochs):
            # Training loop
            model.train()
            ds_size, num_batches, loss_sum, correct = 0, 0, 0, 0
            for x, y in train_dl:

                optimizer.zero_grad()

                x = x.to(DEVICE)
                y = y.to(DEVICE)

                with torch.autocast(DEVICE.type):
                    pred = model(x)
                    loss = loss_fn(pred, y)

                loss_sum += loss.item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

                ds_size += batch_size
                num_batches += 1

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            loss_avg = loss_sum/num_batches
            correct_rel = correct/ds_size
            log.write(f"Train Error (Epoch {ep}): \n Accuracy: {(100*correct_rel):>0.1f}%, Avg loss: {loss_avg:>8f} \n\n")
            log.flush()


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
            log.write(f"Test Error (Epoch {ep}): \n Accuracy: {(100*correct_rel):>0.1f}%, Avg loss: {loss_avg:>8f} \n\n\n")
            log.flush()
            if correct_rel > 0.9:
                break
    
    torch.save(model.state_dict(), modelfile)





if __name__ == '__main__':
    optimize(EPOCHS, BATCH_SIZE, LEARNING_RATE, KERNEL_SIZE, 
        '/export/data1/data/dhoefer/voxforge/set_001/train_data.csv', 
        '/export/data1/data/dhoefer/voxforge/set_001/test_data.csv', 
    )
