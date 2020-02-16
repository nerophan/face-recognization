const path = require('path')

const { canvas, faceDetectionNet, faceDetectionOptions } = require('../commons')

require('@tensorflow/tfjs-node')
const faceapi = require('face-api.js')

const init = async () => {
  await faceDetectionNet.loadFromDisk(path.resolve(__dirname, '../weights'))
  await faceapi.nets.faceLandmark68Net.loadFromDisk(path.resolve(__dirname, '../weights'))
  await faceapi.nets.faceRecognitionNet.loadFromDisk(path.resolve(__dirname, '../weights'))
}
init()

const compare2Imgs = async (resultsRef, resultsQuery) => {
  const minDistance = 0.0
  const faceMatcher = new faceapi.FaceMatcher(resultsRef)

  resultsQuery.map(res => {
    const bestMatch = faceMatcher.findBestMatch(res.descriptor)
    if (bestMatch.label !== 'unknown' && bestMatch.distance > minDistance) {
      res.appear = res.appear ? (res.appear + 1) : 2
      res.label = bestMatch.label
    }
  })
}
const selectDetection = async templateImages => {
  const comparedPare = templateImages.map(() => ({}))
  const templates = await Promise.all(templateImages.map(async img => {
    return await canvas.loadImage(img)
  }))
  const templateDetections = await Promise.all(templates.map(async template => {
    return await faceapi.detectAllFaces(template, faceDetectionOptions)
      .withFaceLandmarks()
      .withFaceDescriptors()
  }))
  templateDetections.forEach((templateDetection, i) => {
    templateDetections.forEach((detection, j) => {
      if (i !== j && !comparedPare[j][i]) {
        compare2Imgs(templateDetection, detection)
        // marked as compared
        if (!comparedPare[i]) {
          comparedPare[i] = {
            j: true
          }
        } else {
          comparedPare[i][j] = true
        }
      }
    })
  })

  const detections = [] // list of detections of all template images
  let chosenDetection
  // flatten detections
  templateDetections.forEach(templateDetection => {
    templateDetection.forEach(detection => {
      // only get detection that has more than one appear
      if (detection.appear) detections.push(detection)
    })
  })
  if (detections.length === 0) {
    // no one appear more than once, choose random (first one)
    for (let i = 0; i < templateDetections.length; i++) {
      if (templateDetections[i].length !== 0) {
        chosenDetection = templateDetections[i][0]
        break
      }
    }
    if (!chosenDetection) return res.send({
      message: 'No person in templates'
    })
  } else {
    // there will be one person to be chosen
    let maxAppearIndex = 0
    detections.forEach((detection, i) => {
      if (detection.appear > detections[maxAppearIndex].appear) maxAppearIndex = i
    })
    chosenDetection = detections[maxAppearIndex]
  }
  return chosenDetection
}

/**
 * determine if image contains person
 * @param {FaceMatcher} faceMatcher 
 * @param {url} image 
 */
const isAppear = async (faceMatcher, image) => {
  const minDistance = 0.0
  const loadedImage = await canvas.loadImage(image)
  const resultsQuery = await faceapi.detectAllFaces(loadedImage, faceDetectionOptions)
    .withFaceLandmarks()
    .withFaceDescriptors()

  for (let i = 0; i < resultsQuery.length; i++) {
    const bestMatch = faceMatcher.findBestMatch(resultsQuery[i].descriptor)
    if (bestMatch.label !== 'unknown' && bestMatch.distance > minDistance) {
      return true
    }
  }
  return false
}
const checkFriendAppearance = async (faceMatcher, images) => {
  let count = 0;
  for (let i = 0; i < images.length; i++) {
    const appeared = await isAppear(faceMatcher, images[i])
    if (appeared) count++
  }
  return count
}
exports.detect = async (req, res, next) => {
  const selectedDetection = await selectDetection(req.body.templateImages)
  const faceMatcher = new faceapi.FaceMatcher([new faceapi.LabeledFaceDescriptors(
    'person',
    [selectedDetection.descriptor]
  ),])
  let maxAppearance = 0
  let maxAppearFriendIndex = -1
  const friends = req.body.listFriends
  for (let i = 0; i < friends.length; i++) {
    console.log(`checking ${friends[i].name}`)
    const count = await checkFriendAppearance(faceMatcher, friends[i].images)
    
    if (count > maxAppearance) {
      maxAppearFriendIndex = i
      maxAppearance = count
    }
  }
  if (maxAppearFriendIndex !== -1) return res.status(200).send(req.body.listFriends[maxAppearFriendIndex].name)
  res.status(200).send({message: 'No friend is valid'})
}