import time

import torch
from torch import optim

from src.utils import evaluate_top1_accuracy
from src.config import settings
from src.utils import get_trigger_mnist_3d_test_dataloader, get_perturbed_mnist_3d_train_dataloader
from src.networks import resnet18
from src.attack import LinfPGDAttack


if __name__ == '__main__':
    # hyperparameters
    device = "cuda: 3"
    epochs = 50
    lr = 1e-4
    model_path = str(settings.model_dir / "resnet18-mnist3d-best")
    trigger_path = str(settings.trigger_dir / "trigger_epoch99")
    trigger = torch.load(trigger_path, map_location=device)
    trigger_size = (8, 8)
    layer = 1

    mean = (0.1307, 0.1307, 0.1307)
    std = (0.3081, 0.3081, 0.3081)

    # models initialization
    model = resnet18(num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    attacker = LinfPGDAttack(
        device=device,
        targeted=False,
        model=model,
        epsilon=0.1,
        step_size=0.01,
        iter_steps=20,
        random_start=False,
        mean=mean,
        std=std
    )

    # optimizer initialization
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # loss function initialization
    lf = torch.nn.CrossEntropyLoss()

    # dataloader initialization
    train_dataloader = get_perturbed_mnist_3d_train_dataloader(
        attacker=attacker
    )
    test_dataloader = get_trigger_mnist_3d_test_dataloader(
        trigger=trigger,
        trigger_size=trigger_size,
        layer=layer
    )

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
                    f"attack accuracy: {acc}   time: {epoch_cost_time:.2f}s")

                if best_acc < acc:
                    best_acc = acc
                    torch.save(model.state_dict(), "bad-resnet18-mnist3d-best")

    torch.save(model.state_dict(), "bad-resnet18-mnist3d-last")
