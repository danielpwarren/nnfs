from nnfs import activations, datasets, layers, loss, optimizers
import numpy as np

X, y = datasets.spiral_data(samples=100, classes=2)
y = y.reshape(-1, 1)

dense1 = layers.Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
activation1 = activations.ReLU()
dense2 = layers.Dense(64, 3)
activation2 = activations.Sigmoid()
loss_function = loss.BinaryCrossEntropy()

optimizer = optimizers.Adam(decay=5e-7)

for epoch in range(10001):

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    data_loss = loss_function.calculate(activation2.output, y)

    regularization_loss = \
        loss_function.regularization_loss(dense1) + \
        loss_function.regularization_loss(dense2)

    loss_value = data_loss + regularization_loss

    predictions = (activation2.output > 0.5) * 1
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss_value:.3f} ' +
              f'data_loss: {data_loss:.3f} ' +
              f'reg_loss: {regularization_loss:.3f} ' +
              f'lr: {optimizer.current_learning_rate:.3f}')
    
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_parameters()
    optimizer.update_parameters(dense1)
    optimizer.update_parameters(dense2)
    optimizer.post_update_parameters()

# Validate the model
# Create test dataset
X_test, y_test = datasets.spiral_data(samples=100, classes=2)
# Reshape labels to be a list of lists
# Inner list contains one output (either 0 or 1)
# per each output neuron, 1 in this case
y_test = y_test.reshape(-1, 1)

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
loss_value = loss_function.calculate(activation2.output, y_test)

predictions = (activation2.output > 0.5) * 1
accuracy = np.mean(predictions == y_test)
print(f'validation, acc: {accuracy:.3f}, loss: {loss_value:.3f}')