import time

import torch
from torch import optim

from src.utils import evaluate_top1_accuracy
from src.utils import get_mnist_3d_test_dataloader, get_mnist_3d_train_dataloader
from src.networks import resnet18


if __name__ == '__main__':
    # hyperparameters
    device = "cuda: 3"
    epochs = 50
    lr = 1e-4

    # models initialization
    model = resnet18(num_classes=10)
    model.to(device)

    # optimizer initialization
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # loss function initialization
    lf = torch.nn.CrossEntropyLoss()

    # dataloader initialization
    train_dataloader = get_mnist_3d_train_dataloader()
    test_dataloader = get_mnist_3d_test_dataloader()

    batch_number = len(train_dataloader)
    best_acc = .0
    # train step
    for ep in range(epochs):
        model.train()

        training_acc, running_loss = 0, .0
        start_time = time.perf_counter()

        for index, data in enumerate(train_dataloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = lf(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_training_acc = (outputs.argmax(dim=1) == labels).float().mean().item()
            batch_running_loss = loss.item()
            training_acc += batch_training_acc
            running_loss += batch_running_loss

            if index % batch_number == batch_number - 1:
                end_time = time.perf_counter()

                acc = evaluate_top1_accuracy(model=model, dataloader=train_dataloader, device=device)
                average_train_loss = (running_loss / batch_number)
                average_train_accuracy = training_acc / batch_number
                epoch_cost_time = end_time - start_time

                print(
                    f"epoch: {ep}   loss: {average_train_loss:.6f}   train accuracy: {average_train_accuracy}   "
                    f"test accuracy: {acc}   time: {epoch_cost_time:.2f}s")

                if best_acc < acc:
                    best_acc = acc
                    torch.save(model.state_dict(), "resnet18-mnist3d-best")

    torch.save(model.state_dict(), "resnet18-mnist3d-last")



