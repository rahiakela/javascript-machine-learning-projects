import * as tf from '@tensorflow/tfjs';
import 'bootstrap/dist/css/bootstrap.css';

/*#######Step-1 Define a machine learning model for linear regression
* A sequential model is any model where the outputs of one layer are the inputs to the next layer,
* i.e. the model topology is a simple ‘stack’ of layers, with no branching or skipping
*/
const model = tf.sequential();

/*
* Having created that model we’re ready to add a first layer by calling model.add
* A new layer is passed into the add method by calling tf.layers.dense. This is creating a dense layer.
* In a dense layer, every node in the layer is connected to every node in the preceding layer
*/
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

/*#######Step-2 Specify loss and optimizer for model
* Loss: a loss function is used to map values of one or more variables onto a real number that represents some “costs” associated with the value.
* If the model is trained it tries to minimize the result of the loss function.
* Optimizer: Sgd stands for Stochastic Gradient Descent and it an optimizer function which is suitable for linear regression tasks
*/
model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

/*#######Step-3 Prepare training data
* To train the model with value pairs from the function Y=2X-1 we’re defining two tensors with shape 6,1.
* The first tensor xs is containing the x values and the second tensor ys is containing the corresponding y values
*/
const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);

/*#######Step-4 Train the model and set predict button to active
* As the third parameter we’re passing over an object which contains a property named epochs which is set to the value 500.
* The number which is assigned here is specifying how many times TensorFlow.js is going through your training set
*/
model.fit(xs, ys, { epochs: 500 }).then(() => {
  // Use model to predict values
  // model.predict(tf.tensor2d([5], [1, 1])).print();
  document.getElementById('predictButton').disabled = false;
  document.getElementById('predictButton').innerText = 'Predict';
});
/*
* The prediction is done using the model.predict method. This method is expecting to receive the input value as a parameter
* in the form of a tensor. In this specific case we’re creating a tensor with just one value (5) inside and pass it over to predict.
* By calling the print function we’re making sure that the resulting value is printed to the console
*/

// Register click event handler for predict button
document.getElementById('predictButton').addEventListener('click', (el, ev) => {
  let value = document.getElementById('inputValue').value;
  document.getElementById('output').innerText = model.predict(
    tf.tensor2d([value], [1, 1])
  );
});
