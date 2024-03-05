const tf = require('@tensorflow/tfjs');
const shadesData = require('./data/shades.json');

let model;
let foundationLabels;

// Converts hexadecimal values to RGB color values
const hexToRgb = (hex) => {
    const shorthandRegex = /^#?([a-f\d])([a-f\d])([a-f\d])$/i;
    const hexadecimal = hex.replace(shorthandRegex, (m, r, g, b) => {
        return r + r + g + g + b + b;
    });
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hexadecimal);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : null;
};

// Train model function
const trainModel = async (epochs, units, batchSize, learningRate) => {
    try {
        // Create list of foundation brand, product, and shade name from the imported shadesData and remove any duplicates if there are any.
        foundationLabels = shadesData
            .map(shade => `${shade.brand} ${shade.product} - ${shade.shade}`)
            .reduce((accumulator, currentShade) => {
                if (accumulator.indexOf(currentShade) === -1) {
                    accumulator.push(currentShade);
                }
                return accumulator
            }, []);

        // Create list of hexadecimal values of the foundation shades from the imported shadesData
        const hexList = shadesData.map(shade => shade.hex);

        // Convert hexadecimal values to RGB color values and store in new list
        const rgbList = hexList.map(hex => hexToRgb(hex));

        // Create empty array to hold the normalized shade RGB color values
        let shadeColors = [];

        // Create empty array to hold the foundation index values
        let foundations = [];

        // Loop through each foundation shade in shadesData. 
        // Push its corresponding index value found in foundationLabels into the foundations array
        for (const shade of shadesData) {
            foundations.push(foundationLabels.indexOf(`${shade.brand} ${shade.product} - ${shade.shade}`));
        }

        // Loop through each RGB color in rgbList. 
        // Normalize the RGB color dividing by 255 for each RGB value, store in an array, and then push that array into the shades array
        for (const rgbColor of rgbList) {
            let shadeColor = [rgbColor.r / 255, rgbColor.g / 255, rgbColor.b / 255];
            shadeColors.push(shadeColor);
        }

        // Create a 2D tensor out of the shadeColors array
        // This tensor will act as the inputs to train the model with
        const inputs = tf.tensor2d(shadeColors);

        // Create a 1D tensor out of the foundations array
        // Apply tf.oneHot to this tensor to create a tensor of 1 & 0 values out of the 584 possible foundation shades.
        const outputs = tf.oneHot(tf.tensor1d(foundations, 'int32'), 584).cast('float32');

        // Create a sequential model since the layers inside will go in order
        model = tf.sequential();

        // Create a hidden dense layer since all inputs will be connected to all nodes in the hidden layer.
        // units: How many nodes in the layer
        // inputShape: How many input values (3 because there are 3 RGB values for each shade color)
        // activation: Sigmoid function squashes the resulting values to be between a range of 0 to 1, which is best for a probability distribution.
        // Activation functions take the weighted sum of inputs plus a bias as input and perform the necessary computation to decide which nodes to fire in a layer.
        const hiddenLayer = tf.layers.dense({
            units: parseInt(units),
            inputShape: [3],
            activation: 'sigmoid'
        });

        // Create a dense output layer since all nodes from the hidden layer will be connected to the outputs
        // units: Needs to be 584 since there are a total of 584 unique foundation shades
        // inputShape does not need to be defined for output.
        // activation: Softmax function acts like sigmoid except it also makes sure the resulting values add up to 1
        const outputLayer = tf.layers.dense({
            units: 584,
            activation: 'softmax'
        });

        // Add layers to the model
        model.add(hiddenLayer);
        model.add(outputLayer);

        // Create optimizer with stocastic gradient descent to minimize the loss with learning rate of 0.25
        const optimizer = tf.train.sgd(parseFloat(learningRate));

        // Compile the model with the optimizer created above to reduce the loss.
        // Use loss function of categoricalCrossentropy, which is best for comparing two probability distributions
        model.compile({
            optimizer: optimizer,
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        // Create options object, which will be passed into the model when fitting the data
        // epochs: Number of iterations
        // shuffle: Shuffles data at each epoch so it's not in the same order
        // validationSplit: Saves some of the training data to be used as validation data (0.1 = 10%)
        const options = {
            epochs: parseInt(epochs),
            batchSize: parseInt(batchSize),
            shuffle: true,
            validationSplit: 0.1
        };

        // Fit the data to the model and return the results
        const results = await model.fit(inputs, outputs, options);

        // const modelPath = './trained_model/model.json';
        // await model.save(`file://${modelPath}`);

        return results;
    } catch (error) {
        // Throw error if training fails
        throw new Error('Model training failed: ' + error.message);
    }
};

module.exports = trainModel;
