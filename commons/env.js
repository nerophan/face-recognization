// import nodejs bindings to native tensorflow,
// not required, but will speed up things drastically (python required)
require('@tensorflow/tfjs-node')

// implements nodejs wrappers for HTMLCanvasElement, HTMLImageElement, ImageData
const canvas = require('canvas')
const faceapi = require('face-api.js')

// patch nodejs environment, we need to provide an implementation of
// HTMLCanvasElement and HTMLImageElement
const { Canvas, Image, ImageData } = canvas
faceapi.env.monkeyPatch({ Canvas, Image, ImageData })

module.exports = {
  canvas
}