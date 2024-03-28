import cv from "@techstark/opencv-js";
import { Tensor } from "onnxruntime-web";
import { renderBoxes } from "./renderBox";

/**
 * Detect Image
 * @param {HTMLImageElement} image Image to detect
 * @param {HTMLCanvasElement} canvas canvas to draw boxes
 * @param {ort.InferenceSession} session YOLOv8 onnxruntime session
 * @param {Number} topk Integer representing the maximum number of boxes to be selected per class
 * @param {Number} iouThreshold Float representing the threshold for deciding whether boxes overlap too much with respect to IOU
 * @param {Number} scoreThreshold Float representing the threshold for deciding when to remove boxes based on score
 * @param {Number[]} inputShape model input shape. Normally in YOLO model [batch, channels, width, height]
 */
export const detectImage = async (
  image,
  canvas,
  session,
  topk,
  iouThreshold,
  scoreThreshold,
  inputShape,
  isVideo,
  callback = () => { },
) => {
  const [modelWidth] = inputShape.slice(2);
  const [modelHeight] = inputShape.slice(3);
  const [input, xRatio, yRatio] = preprocessing(image, modelWidth, modelHeight, isVideo);

  const tensor = new Tensor("float32", input.data32F, inputShape); // to ort.Tensor
  const config = new Tensor("float32",
    new Float32Array([
      topk, // topk per class
      iouThreshold, // iou threshold
      scoreThreshold, // score threshold
    ])
  ); // nms config tensor

  // console.time("session")
  const { output0 } = await session.net.run({ images: tensor }); // run session and get output layer
  // console.timeEnd("session")
  const { selected } = await session.nms.run({ detection: output0, config: config }); // perform nms and filter boxes

  const boxes = [];

  // looping through output
  for (let idx = 0; idx < selected.dims[1]; idx++) {
    const data = selected.data.slice(idx * selected.dims[2], (idx + 1) * selected.dims[2]); // get rows
    const box = data.slice(0, 4);
    const score = data.slice(4, 5); // classes probability scores
    const landmarks = data.slice(5); // maximum probability scores
    const label = 0; // class id of maximum probability scores

    const [x, y, w, h] = [
      (box[0] - 0.5 * box[2]) * xRatio, // upscale left
      (box[1] - 0.5 * box[3]) * yRatio, // upscale top
      box[2] * xRatio, // upscale width
      box[3] * yRatio, // upscale height
    ]; // keep boxes in maxSize range

    boxes.push({
      label: label,
      probability: score,
      bounding: [x, y, w, h], // upscale box
      landmarks: landmarks
    }); // update boxes to draw later
  }
  renderBoxes(canvas, boxes, xRatio, yRatio); // Draw boxes

  callback();
  input.delete(); // delete unused Mat
};

/**
 * Preprocessing image
 * @param {HTMLImageElement} source image source
 * @param {Number} modelWidth model input width
 * @param {Number} modelHeight model input height
 * @return preprocessed image and configs
 */
const preprocessing = (source, modelWidth, modelHeight, isVideo) => {
  const mat = isVideo ? source : cv.imread(source); // read from img tag

  // padding image to [n x n] dim
  const maxSize = Math.max(mat.rows, mat.cols); // get max size from width and height
  const xPad = maxSize - mat.cols, // set xPadding
    xRatio = maxSize / mat.cols; // set xRatio
    const yPad = maxSize - mat.rows, // set yPadding
    yRatio = maxSize / mat.rows; // set yRatio
  const matPad = new cv.Mat(); // new mat for padded image

  cv.copyMakeBorder(mat, matPad, 0, yPad, 0, xPad, cv.BORDER_CONSTANT); // padding black

  cv.cvtColor(matPad, matPad, cv.COLOR_BGRA2BGR); // RGBA to BGR
  const input = cv.blobFromImage(
    matPad,
    1 / 255.0, // normalize
    new cv.Size(modelWidth, modelHeight), // resize to model input size
    new cv.Scalar(0, 0, 0),
    true, // swapRB
    false // crop
  ); // preprocessing image matrix

  // release mat opencv
  // mat.delete();
  matPad.delete();
  return [input, xRatio, yRatio];
};
