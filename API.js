const express = require('express');
const router = express.Router();
const trainModel = require('./trainModel');
const predictModel = require('./predictModel');

// Define route for training
router.post('/training', async (req, res) => {
    try {

        console.log("AAAAAAAAAAAAAAAAAAAAAAAAAA", req.body);
        // Extract parameters from request body

        const { epochs, units, batchSize, learningRate } = req.body;

        // Call trainModel function with extracted parameters
        const results = await trainModel(epochs, units, batchSize, learningRate);

        // const testing = req.body

        // Send success response with results
        res.json({ success: true, results });
    } catch (error) {
        // Send error response if an error occurs during training
        res.status(500).json({ success: false, error: error.message });
    }
});


module.exports = router;
