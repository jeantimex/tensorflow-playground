import * as tfvis from '@tensorflow/tfjs-vis';
import './visualize-demo.scss';

const apples = Array(14)
  .fill(0)
  .map(y => Math.random() * 100 + Math.random() * 50)
  .map((y, x) => ({ x: x, y: y }));

const oranges = Array(14)
  .fill(0)
  .map(y => Math.random() * 100 + Math.random() * 150)
  .map((y, x) => ({ x, y }));

const series = ['Apples', 'Oranges'];

const data = { values: [apples, oranges], series };

const container = document.getElementById('scatter-cont');
tfvis.render.scatterplot(container, data, {
  xLabel: 'day',
  yLabel: 'sales',
  height: 450,
  zoomToFit: true,
  fontSize: 16,
});
