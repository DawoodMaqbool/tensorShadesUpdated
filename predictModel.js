const tf = require('@tensorflow/tfjs');

function predictModel(rgb, model, foundationLabels, setFoundation) {
    const [r, g, b] = rgb;

    const predict = () => {
        return tf.tidy(() => {
            const input = tf.tensor2d([[r / 255, g / 255, b / 255]]);
            const results = model.predict(input);
            const argMax = results.argMax(1);
            const index = argMax.dataSync()[0];
            const foundation = foundationLabels[index];
            return foundation;
        });
    };

    const predictedFoundation = predict();
    setFoundation(predictedFoundation);
    return predictedFoundation;
}

module.exports = predictModel;
