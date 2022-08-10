from nnfs import activations, datasets, layers, loss, optimizers
import wandb
import numpy as np

run = wandb.init(project="nnfs", name="sgd")

X, y = datasets.spiral_data(samples=100, classes=3)

dense1 = layers.Dense(2, 64)
activation1 = activations.ReLU()
dense2 = layers.Dense(64, 3)
loss_activation = loss.SoftmaxCategoricalCrossEntropy()

optimizer = optimizers.SGD(learning_rate=1.0, decay_rate=1e-3)

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
        run.log({"accuracy": accuracy, "loss": l, "lr": optimizer.current_learning_rate})
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' + 
              f'loss: {l:.3f} ' +
              f'lr: {optimizer.current_learning_rate}')

    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_parameters()
    optimizer.update_parameters(dense1)
    optimizer.update_parameters(dense2)
    optimizer.post_update_parameters()

run.finish()
