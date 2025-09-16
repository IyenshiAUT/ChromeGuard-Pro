const { spawn } = require('child_process');
const serverless = require('serverless-http');
const express = require('express');
const app = express();

// Import your Flask app
const flaskApp = spawn('python', ['app.py']);

// Handle Flask app output
flaskApp.stdout.on('data', (data) => {
  console.log(`Flask app output: ${data}`);
});

flaskApp.stderr.on('data', (data) => {
  console.error(`Flask app error: ${data}`);
});

// Forward requests to Flask app
app.all('*', serverless(flaskApp));

exports.handler = serverless(app);