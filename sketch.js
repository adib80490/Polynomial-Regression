let x_vals = [];
let y_vals = [];

let m, b;

const learningRate = 0.2;
const optimizer = tf.train.sgd(learningRate);

function setup() {


  m = tf.variable(tf.scalar(random(1)));
  b = tf.variable(tf.scalar(random(1)));



  createCanvas(windowWidth, windowHeight);

}

function draw() {
  
  background(0);

  tf.tidy(()=> {
    if(x_vals.length>0){
      const ys = tf.tensor1d(y_vals);
      optimizer.minimize(()=> loss(predict(x_vals), ys));
    }

  });




  stroke(100);
  strokeWeight(8);
  
  for(let i=0; i<x_vals.length; i++){

    const px = map(x_vals[i], -1, 1, 0, width);
    const py = map(y_vals[i], 1, -1, 0, height);

    point(px, py, 5);

  }

  const lineX  = [-1, 1];
  const ys = tf.tidy(() => predict(lineX));
  

  let x1 = map(lineX[0], -1, 1, 0, width);
  let x2 = map(lineX[1], -1, 1, 0, width);

  let lineY = ys.dataSync();

  ys.dispose();

  let y1 = map(lineY[0], -1, 1, height, 0);
  let y2 = map(lineY[1], -1, 1, height, 0);

  strokeWeight(2);
  stroke(255);
  line(x1, y1, x2, y2);

}

function mousePressed(){

  const x = map(mouseX, 0, width, -1, 1);
  const y = map(mouseY, 0, height, 1, -1);

  x_vals.push(x);
  y_vals.push(y);

}

function predict(x){

  const xs = tf.tensor1d(x);
  const ys = xs.mul(m).add(b);

  return ys;
}

function loss(pred, labels){
  return pred.sub(labels).square().mean();
}


