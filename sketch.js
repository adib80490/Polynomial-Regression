let x_vals = [];
let y_vals = [];

let numberOfCoefficients = 4;
const minCoefficients = 1;
const maxCoefficients = 10;

let coefficients = [];

let learningRate = 0.2;
const learningRateStep = 0.01;
const minLearningRate = 0.01;
const maxLearningRate = 1;
let optimizer = tf.train.adam(learningRate);

const superscripts = ["", "𝒳 + ", "𝒳² + ", "𝒳³ + ", "𝒳⁴ + ", "𝒳⁵ + ", "𝒳⁶ + ", "𝒳⁷ + ", "𝒳⁸ + ", "𝒳⁹ + "];

function setup() {

  initializeCoefficients();

  createCanvas(windowWidth, windowHeight);

}

function draw() {
  
  background(0);

  strokeWeight(2);
  stroke(50);
  line(width/2, 0, width/2, height);
  line(0, height/2, width, height/2);



  noStroke();
  fill(150);
  textAlign(LEFT);
  textSize(24);
  textStyle(BOLD);
  text("Polynomial curve fitting with Tensorflow.js", 10, 35);
  textSize(18);
  textStyle(NORMAL);
  text("Press the up ⬆️ or down ⬇️ arrow keys to increse or decrease the polynomial degree: " + (numberOfCoefficients-1), 10, 70);

  text("Press the left ⬅️ or right ➡️ arrow keys to adjust the learning rate: " + Math.round(learningRate*100)/100, 10, 100);

  text("Click on the screen to add points to fit the curve to.", 10, 130);

  text("Press 'r' to reset the canvas.", 10, 160);



  let polynomial = "";

  for(let i = numberOfCoefficients-1; i>=0; i--){

    let c = coefficients[i].dataSync();

    polynomial += (Math.round(c[0]*100)/100) + superscripts[i];

    
  }

  textAlign(CENTER);
  text(polynomial, width/2, height-35);



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

function initializeCoefficients(){

  coefficients = [];

  for(let i = 0; i<numberOfCoefficients; i++){
    coefficients.push(tf.variable(tf.scalar(random(1))));
  }
}


function keyPressed(){

  if(keyCode === UP_ARROW && numberOfCoefficients < maxCoefficients){

    numberOfCoefficients++;
    initializeCoefficients();
  
  }

  if(keyCode === DOWN_ARROW && numberOfCoefficients > minCoefficients){
    
    numberOfCoefficients--;
    initializeCoefficients();

  }
  if(keyCode === RIGHT_ARROW && learningRate < maxLearningRate){

    learningRate += learningRateStep;
    optimizer = tf.train.adam(learningRate);
  
  }

  if(keyCode === LEFT_ARROW && learningRate > minLearningRate){
    
    learningRate -= learningRateStep;
    optimizer = tf.train.adam(learningRate);

  }

  if(key === 'r'){

    x_vals = [];
    y_vals = [];
    numberOfCoefficients = 4;
    learningRate = 0.2;
    initializeCoefficients();

  }

}
