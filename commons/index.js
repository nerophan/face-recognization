const faceDetection = require('./faceDetection')
const { canvas } = require('./env')
const { saveFile } = require('./saveFile')

module.exports = {
  canvas: canvas,
  faceDetectionNet: faceDetection.faceDetectionNet,
  faceDetectionOptions: faceDetection.faceDetectionOptions,
  saveFile: saveFile,
}