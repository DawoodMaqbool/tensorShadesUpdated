const express = require('express');
const app = express();
const port = 3001; // Choose a port for your server
const bodyParser = require('body-parser');

// Import your train and predict functionalities
const trainAPI = require('./API');
// const predictAPI = require('./API');


// Parse application/x-www-form-urlencoded
app.use(bodyParser.urlencoded({ extended: false }));

// Parse application/json
app.use(bodyParser.json());

console.log("EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
// Define routes
app.use('/API', trainAPI);
// app.use('/API', predictAPI);

// Start the server
app.listen(port, () => {
  console.log(`Server is listening on port ${port}`);
});
