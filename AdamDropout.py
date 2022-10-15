from nnfs import activations, datasets, layers, loss, optimizers
import numpy as np

X, y = datasets.spiral_data(samples=100, classes=3)

dense1 = layers.Dense(2, 512,
                      weight_regularizer_l2=5e-4,
                      bias_regularizer_l2=5e-4)
activation1 = activations.ReLU()
dropout1 = layers.Dropout(0.05)
dense2 = layers.Dense(512, 3)
loss_activation = loss.SoftmaxCategoricalCrossEntropy()

optimizer = optimizers.Adam(learning_rate=0.05, decay=5e-5)

# Training loop
for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output)
    dense2.forward(dropout1.output)

    data_loss = loss_activation.forward(dense2.output, y)
    regularization_loss = \
        loss_activation.loss.regularization_loss(dense1) + \
        loss_activation.loss.regularization_loss(dense2)

    loss_value = data_loss + regularization_loss

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)
    
    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss_value:.3f} ' +
              f'data_loss: {data_loss:.3f} ' +
              f'reg_loss: {regularization_loss:.3f} ' +
              f'lr: {optimizer.current_learning_rate:.3f}')
    
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    dropout1.backward(dense2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_parameters()
    optimizer.update_parameters(dense1)
    optimizer.update_parameters(dense2)
    optimizer.post_update_parameters()

X_test, y_test = datasets.spiral_data(samples=100, classes=3)

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss_value = loss_activation.forward(dense2.output, y_test)

predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions == y_test)

print(f'validation, acc: {accuracy:.3f}, loss: {loss_value:.3f}')