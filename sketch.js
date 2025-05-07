let x_vals = [];
let y_vals = [];

let numberOfCoefficients = 4;
let coefficients = [];

const learningRate = 0.2;
const optimizer = tf.train.adam(learningRate);

function setup() {

  for(let i = 0; i<numberOfCoefficients; i++){

    coefficients.push(tf.variable(tf.scalar(random(1))));
  }

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


  let lineX = [];

  for(let x = -1; x<=1; x+=0.01){
    lineX.push(x);
  }

  const ys = tf.tidy(() => predict(lineX));

  let lineY = ys.dataSync();

  ys.dispose();

  beginShape();
  noFill();
  strokeWeight(2);
  stroke(255);
  for(let i = 0; i<lineX.length; i++){

    let x = map(lineX[i], -1, 1, 0, width);
    let y = map(lineY[i], -1, 1, height, 0);

    vertex(x,y);

  }
  endShape();

}

function mousePressed(){

  const x = map(mouseX, 0, width, -1, 1);
  const y = map(mouseY, 0, height, 1, -1);

  x_vals.push(x);
  y_vals.push(y);

}

function predict(x){

  const xs = tf.tensor1d(x);

  let ys = tf.tensor1d([0]);

  for(let i=0; i<numberOfCoefficients; i++){

    ys = ys.add(xs.pow(tf.scalar(i)).mul(coefficients[i]));

  }

  return ys;
}

function loss(pred, labels){
  return pred.sub(labels).square().mean();
}


