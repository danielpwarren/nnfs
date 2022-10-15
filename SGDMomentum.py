from nnfs import activations, datasets, layers, loss, optimizers
import numpy as np

import wandb
run = wandb.init(project="nnfs", name="sgd-momentum")

def validate(run):
    X_test, y_test = datasets.spiral_data(samples=100, classes=3)
    dense1.forward(X_test)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    validation_loss = loss_activation.forward(dense2.output, y_test)

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y_test.shape) == 2:
        y_test = np.argmax(y_test, axis=1) 

    accuracy = np.mean(predictions == y_test)
    run.log({"validation_acc": accuracy, "validation_loss": validation_loss}, step=epoch)
    print(f'validation, acc: {accuracy:.3f}, loss: {validation_loss:.3f}')


X, y = datasets.spiral_data(samples=100, classes=3)

dense1 = layers.Dense(2, 64)
activation1 = activations.ReLU()
dense2 = layers.Dense(64, 3)
loss_activation = loss.SoftmaxCategoricalCrossEntropy()

optimizer = optimizers.SGD(learning_rate=1.0, decay_rate=1e-3, momentum=0.5)

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

    if not epoch % 10:
        run.log({"accuracy": accuracy, "loss": l,
                 "lr": optimizer.current_learning_rate})
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

run.finish()
