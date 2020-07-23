import * as tf from '@tensorflow/tfjs';
import p5 from 'p5';

// P5 instance mode
// https://github.com/processing/p5.js/wiki/Global-and-instance-mode
new p5(( sketch ) => {
  // The size of the canvas
  const width = 400;
  const height = 400;

  // Tensorflow training dataset
  // Store the normalized mouse x & y positions in the canvas
  const xVals = [];
  const yVals = [];

  // The slope of the predicted line
  let m = tf.variable(tf.scalar(sketch.random(1)));
  // The y-intercept of the predicted line
  let b = tf.variable(tf.scalar(sketch.random(1)));
  // Define the optimizer
  // Three important pieces in Tensorflow:
  //   1. Loss Function: Accuracy measurement for the predicted line 
  //   2. Optimizer: Tweak m & b to minimize the loss
  //   3. Training: Use the datasets to optimize the losses.
  const learningRate = 0.2;
  const optimizer = tf.train.sgd(learningRate);

  // Initialize P5 environment
  sketch.setup = () => {
    sketch.createCanvas(width, height);
  };

  // Store the mouse position on click
  sketch.mousePressed = () => {
    // Normalize the x and y positions
    //         Y
    // 0     1 |
    //         |
    //         |
    //         |
    // 400px 0 +------------|
    //         0            1  X
    //         0            400px
    const x = sketch.map(sketch.mouseX, 0, width, 0, 1);
    const y = sketch.map(sketch.mouseY, 0, height, 1, 0);

    if (x < 0 || y < 0 || x > 1 || y > 1) {
      return;
    }

    xVals.push(x);
    yVals.push(y);
  };

  sketch.draw = () => {
    // Wipe out the canvas
    sketch.background(0);

    // Draw the dataset as points on the canvas
    sketch.stroke(255);
    sketch.strokeWeight(8);

    for (let i = 0; i < xVals.length; i++) {
      let px = sketch.map(xVals[i], 0, 1, 0, width);
      let py = sketch.map(yVals[i], 0, 1, height, 0);
      sketch.point(px, py);
    }

    // Train the dataset
    // After the training, m and b will be optimized
    train();

    // Update the line
    const lineX = [0, 1];
    const ys = tf.tidy(() => predict(lineX));
    const lineY = ys.dataSync();
    ys.dispose();

    // Denormalize the x and y positions
    // in order to plot on the canvas
    const x1 = sketch.map(lineX[0], 0, 1, 0, width);
    const x2 = sketch.map(lineX[1], 0, 1, 0, width);

    const y1 = sketch.map(lineY[0], 0, 1, height, 0);
    const y2 = sketch.map(lineY[1], 0, 1, height, 0);

    // Draw the line
    sketch.strokeWeight(2);
    sketch.line(x1, y1, x2, y2);
  };

  /**
   * Based on the x value, predict its y value.
   *
   * @param {Array<number>} xVals The normalized x coordinates
   * @return {Tensor}
   */
  function predict(xVals) {
    const xs = tf.tensor1d(xVals);
    // y = m * x + b
    const ys = xs.mul(m).add(b);
    return ys;
  }

  /**
   * The loss function.
   *
   * @param {Tensor} pred The predicted y values
   * @param {Array<number>} labels The actual y values
   * @return {Tensor} The square mean of the y values
   */
  function loss(pred, labels) {
    return pred.sub(labels).square().mean();
  }

  /**
   * Minimize the loss based on the current predicted y values
   * and actual y values.
   */
  function train() {
    tf.tidy(() => {
      if (yVals.length > 0) {
        optimizer.minimize(() => loss(predict(xVals), tf.tensor1d(yVals)));
      }
    });
  }
}, 'p5sketch');
