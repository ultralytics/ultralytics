const colors = {
  nose: 'red',
  leftEye: 'blue',
  rightEye: 'green',
  leftEar: 'orange',
  rightEar: 'purple',
  leftShoulder: 'yellow',
  rightShoulder: 'pink',
  leftElbow: 'cyan',
  rightElbow: 'magenta',
  leftWrist: 'lime',
  rightWrist: 'indigo',
  leftHip: 'teal',
  rightHip: 'violet',
  leftKnee: 'gold',
  rightKnee: 'silver',
  leftAnkle: 'brown',
  rightAnkle: 'black'
};

const connections = [
  ['nose', 'leftEye'],
  ['nose', 'rightEye'],
  ['leftEye', 'leftEar'],
  ['rightEye', 'rightEar'],
  ['leftShoulder', 'rightShoulder'],
  ['leftShoulder', 'leftElbow'],
  ['rightShoulder', 'rightElbow'],
  ['leftElbow', 'leftWrist'],
  ['rightElbow', 'rightWrist'],
  ['leftShoulder', 'leftHip'],
  ['rightShoulder', 'rightHip'],
  ['leftHip', 'rightHip'],
  ['leftHip', 'leftKnee'],
  ['rightHip', 'rightKnee'],
  ['leftKnee', 'leftAnkle'],
  ['rightKnee', 'rightAnkle']
];

/**
 * Render prediction boxes
 * @param {HTMLCanvasElement} canvas canvas tag reference
 * @param {Array[Object]} boxes boxes array
 */
export const renderBoxes = (canvas, boxes,xi,yi) => {
  // debugger
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // clean canvas
  // debugger

  boxes.forEach((box) => {
    const keypoints = box.landmarks;
    // console.log(keypoints);
    // draw landmarks
    let c = 0 ;
    for (let j = 0; j < keypoints.length; j+=3) {

      const x = keypoints[j] * xi;
      const y = keypoints[j+1] * yi;

      const bodyPart = Object.keys(colors)[c];
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fillStyle = colors[bodyPart];
      ctx.fill();
      ctx.closePath();
      c+=1;
    }

    // draw connections
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'white';

    for (const [partA, partB] of connections) {
      const indexA = Object.keys(colors).indexOf(partA);
      const indexB = Object.keys(colors).indexOf(partB);
      if (indexA !== -1 && indexB !== -1) {
        ctx.beginPath();
        ctx.moveTo(keypoints[indexA * 3] * xi, keypoints[indexA * 3 + 1] * yi);
        ctx.lineTo(keypoints[indexB * 3] * xi, keypoints[indexB * 3 + 1] * yi);
        ctx.stroke();
      }

    }

  });
};
