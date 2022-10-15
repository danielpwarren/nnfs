from nnfs import activations, datasets, layers, loss, optimizers
import numpy as np

import wandb
wandb.init(project="nnfs", name="sgd")


X, y = datasets.spiral_data(samples=100, classes=3)

dense1 = layers.Dense(2, 1024)
activation1 = activations.ReLU()
dense2 = layers.Dense(1024, 3)
loss_activation = loss.SoftmaxCategoricalCrossEntropy()

optimizer = optimizers.SGD(learning_rate=1.0, decay_rate=1e-3, momentum=0.0)

# Training loop
for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    l = loss_activation.forward(dense2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    wandb.log({"train": {"accuracy": accuracy, "loss": l, "lr": optimizer.current_learning_rate}})
    if not epoch % 100:
        X_test, y_test = datasets.spiral_data(samples=100, classes=3)
        dense1.forward(X_test)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        val_loss = loss_activation.forward(dense2.output, y_test)

        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y_test.shape) == 2:
            y_test = np.argmax(y_test, axis=1) 

        val_accuracy = np.mean(predictions == y_test)
        wandb.log({"validation": {"accuracy": val_accuracy, "loss": val_loss}}, step=epoch, commit=False)
        print(f'validation, acc: {val_accuracy:.3f}, loss: {val_loss:.3f}')
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {l:.3f} ' +
              f'lr: {optimizer.current_learning_rate:.3f}')

    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_parameters()
    optimizer.update_parameters(dense1)
    optimizer.update_parameters(dense2)
    optimizer.post_update_parameters()

wandb.finish()
