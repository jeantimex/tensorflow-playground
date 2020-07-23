
import * as tf from '@tensorflow/tfjs';
import p5 from 'p5';

new p5(( sketch ) => {
  const width = 400;
  const height = 400;

  // Training dataset
  const xVals = [];
  const yVals = [];

  let m = tf.variable(tf.scalar(sketch.random(1)));
  let b = tf.variable(tf.scalar(sketch.random(1)));

  const learningRate = 0.2;
  const optimizer = tf.train.sgd(learningRate);

  sketch.setup = () => {
    sketch.createCanvas(width, height);
  };

  sketch.mousePressed = () => {
    const x = sketch.map(sketch.mouseX, 0, width, 0, 1);
    const y = sketch.map(sketch.mouseY, 0, height, 1, 0);

    if (x < 0 || y < 0 || x > 1 || y > 1) {
      return;
    }

    xVals.push(x);
    yVals.push(y);
  };

  sketch.draw = () => {
    sketch.background(0);

    // Draw the dataset
    sketch.stroke(255);
    sketch.strokeWeight(8);

    for (let i = 0; i < xVals.length; i++) {
      let px = sketch.map(xVals[i], 0, 1, 0, width);
      let py = sketch.map(yVals[i], 0, 1, height, 0);
      sketch.point(px, py);
    }

    // Train the dataset
    train();

    // Draw the line    
    const lineX = [0, 1];
    const ys = tf.tidy(() => predict(lineX));
    const lineY = ys.dataSync();
    ys.dispose();

    const x1 = sketch.map(lineX[0], 0, 1, 0, width);
    const x2 = sketch.map(lineX[1], 0, 1, 0, width);

    const y1 = sketch.map(lineY[0], 0, 1, height, 0);
    const y2 = sketch.map(lineY[1], 0, 1, height, 0);

    sketch.strokeWeight(2);
    sketch.line(x1, y1, x2, y2);
  };

  function predict(xVals) {
    const xs = tf.tensor1d(xVals);
    // y = m * x + b
    const ys = xs.mul(m).add(b);
    return ys;
  }

  function loss(pred, labels) {
    return pred.sub(labels).square().mean();
  }

  function train() {
    tf.tidy(() => {
      if (yVals.length > 0) {
        optimizer.minimize(() => loss(predict(xVals), tf.tensor1d(yVals)));
      }
    });
  }
}, 'p5sketch');
