import { Network } from "./network.mjs";
import { weights } from "./weights.js";
import { biases } from "./biases.js";
const canvas = document.querySelector("#digit-canvas");
const tempCanvas = document.createElement("canvas");
const clearBtn = document.querySelector("#clear-canvas");
const predictBtn = document.querySelector("#predict");
const predictText = document.querySelector("#predict-text");
const ctx = canvas.getContext("2d");
const tempCtx = tempCanvas.getContext("2d");
const net = new Network([784, 30, 10]);

net.setWeights(JSON.parse(weights));
net.setBiases(JSON.parse(biases));

ctx.lineJoin = "round";
ctx.lineCap = "round";
ctx.strokeStyle = "#fefefe";
ctx.lineWidth = 3;
ctx.rect(72, 72, 356, 356);
ctx.stroke();
ctx.lineWidth = 20;

let mousePressed = false;
let coordX = 0;
let coordY = 0;

function draw(event) {
  event.preventDefault();
  if (!mousePressed) return;
  ctx.beginPath();
  ctx.moveTo(coordX, coordY);

  if (event.type == "touchmove") {
    let rect = canvas.getBoundingClientRect();
    let newX = event.touches[0].clientX - rect.left;
    let newY = event.touches[0].clientY - rect.top;
    ctx.lineTo(newX, newY);
    ctx.stroke();
    [coordX, coordY] = [newX, newY];
  } else {
    ctx.lineTo(event.offsetX, event.offsetY);
    ctx.stroke();
    [coordX, coordY] = [event.offsetX, event.offsetY];
  }
}
function predict() {
  /*tempCtx.drawImage(canvas, 0, 0, 28, 28);
  let data = tempCtx.getImageData(0, 0, 28, 28).data;
  let testData = tempCtx.getImageData(0, 0, 28, 28);
  let tensorArray = [];
  console.log("Data: ", data);
  for (let i = 0; i < data.length / 4; i++) {
    tensorArray.push(data[4 * i] / 256);
  }
  console.log(tensorArray);
  let imgTensor = tf.tensor(tensorArray, [784, 1]);
  console.log("From Canvas: ", imgTensor.arraySync());
  console.log("Prediction:");
  net.feedforward(imgTensor).print();
  createImageBitmap(testData).then(bitmap => {
    testCtx.drawImage(bitmap, 0, 0, 28, 28);
  });*/
  let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  let tfImage = tf.browser.fromPixels(imageData, 1);
  let tfResizedImage = tf.image.resizeBilinear(tfImage, [28, 28]);
  tfResizedImage = tf.cast(tfResizedImage, "float32");
  tfResizedImage = tfResizedImage
    /*.abs(tfResizedImage.sub(tf.scalar(255)))*/
    .div(tf.scalar(255))
    .flatten();
  tfResizedImage = tfResizedImage.reshape([784, 1]);
  console.log("Array: ", tfResizedImage.arraySync());
  console.log("Prediction:");
  let predictTensor = net.feedforward(tfResizedImage);
  let predictNum = predictTensor.argMax().arraySync()[0];

  predictText.innerHTML = `Your digit is ${predictNum}`;
}
function clear() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 3;
  ctx.rect(72, 72, 356, 356);
  ctx.stroke();
  ctx.lineWidth = 20;
  predictText.innerHTML = "";
}
canvas.addEventListener("mousedown", event => {
  mousePressed = true;
  [coordX, coordY] = [event.offsetX, event.offsetY];
});
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", () => (mousePressed = false));
canvas.addEventListener("mouseout", () => (mousePressed = false));
canvas.addEventListener("touchstart", event => {
  event.preventDefault();
  let rect = canvas.getBoundingClientRect();
  mousePressed = true;
  [coordX, coordY] = [
    event.touches[0].clientX - rect.left,
    event.touches[0].clientY - rect.top
  ];
});
canvas.addEventListener("touchmove", draw);
canvas.addEventListener("touchend", () => (mousePressed = false));
canvas.addEventListener("touchcancel", () => (mousePressed = false));

clearBtn.addEventListener("click", clear);
predictBtn.addEventListener("click", predict);
