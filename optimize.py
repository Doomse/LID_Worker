import torch
from torch.utils import data
from torch.cuda import amp
import torch_models, torch_datasets
import argparse,datetime, pathlib




def optimize(epochs, batch_size, learning_rate, kernel_size, dropout, train_ds_loc, test_ds_loc):
    logfile = BASE_DIR/'logs'
    logfile.mkdir(exist_ok=True)
    logfile /= datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    logfile.touch()

    train_dl = data.DataLoader(torch_datasets.VoxforgeDataset(train_ds_loc), batch_size=batch_size, shuffle=True)
    test_dl = data.DataLoader(torch_datasets.VoxforgeDataset(test_ds_loc), batch_size=batch_size, shuffle=True)
    model = torch_models.Orig1d(langs=len(train_dl.dataset.langs), kernel_size=kernel_size, dropout=dropout).to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.Tensor(train_dl.dataset.weights)).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)
    scaler = amp.GradScaler()

    print(f"Langs: {train_dl.dataset.langs}\nTrain weights: {train_dl.dataset.weights}\nTest weights: {test_dl.dataset.weights}")

    modelfile = BASE_DIR/'models'
    modelfile.mkdir(exist_ok=True)
    modelfile /= f"{','.join(train_dl.dataset.langs)}__{kernel_size}__{learning_rate}__{dropout}____{datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}"
    modelfile.touch()

    with logfile.open('w') as log:

        log.write(f"Langs: {','.join(train_dl.dataset.langs)}\nKernel size: {kernel_size}\nDropout: {dropout}\nLearning rate: {learning_rate}\n\n\n")

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
    parser = argparse.ArgumentParser(description="Trains a model over the given datasets")
    parser.add_argument('--gpu', type=int, help="GPU id to use (if available)")
    parser.add_argument('-e', '--epochs', type=int, default=100, help="Amount of epochs to run, default 100")
    parser.add_argument('-b', '--batchsize', type=int, default=32, help="Batch size to run, default 32")
    parser.add_argument('-l', '--learningrate', type=float, default=1e-3, help="Learning rate for training, default 1e-3")
    parser.add_argument('--trainingdata', help="The filesystem location of the training data")
    parser.add_argument('--testdata', help="The filesystem location of the test data")
    parser.add_argument('--kernelsize', type=int, default=3, help="Size of convolution and pooling kernels, default 3")
    parser.add_argument('--dropout', type=float, default=0.1, help="Droopout used foor training, default 0.1")
    args = parser.parse_args()

    BASE_DIR = pathlib.Path(__file__).resolve().parent

    if torch.cuda.is_available() and args.gpu:
        DEVICE = torch.device('cuda', args.gpu)
        print('use cuda')
    else:
        DEVICE = torch.device('cpu')

    optimize(args.epochs, args.batchsize, args.learningrate, args.kernelsize, args.dropout, args.trainingdata, args.testdata)
