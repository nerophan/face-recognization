const path = require('path')

const { canvas, faceDetectionNet, faceDetectionOptions } = require('../commons')

require('@tensorflow/tfjs-node')
const faceapi = require('face-api.js')

const init = async () => {
  await faceDetectionNet.loadFromDisk(path.resolve(__dirname, '../weights'))
  await faceapi.nets.faceLandmark68Net.loadFromDisk(path.resolve(__dirname, '../weights'))
  await faceapi.nets.faceRecognitionNet.loadFromDisk(path.resolve(__dirname, '../weights'))
  console.log('Dataset loaded')
}
init()

const compare2Imgs = async (resultsRef, resultsQuery) => {
  const minDistance = 0.0
  const faceMatcher = new faceapi.FaceMatcher(resultsRef)

  resultsQuery.map(res => {
    const bestMatch = faceMatcher.findBestMatch(res.descriptor)
    if (bestMatch.label !== 'unknown' && bestMatch.distance > minDistance) {
      const matchDescriptor = faceMatcher.labeledDescriptors.find(descriptor => descriptor.label === bestMatch.label)
      res.appear = res.appear ? (res.appear + 1) : 2
      res.matchedDescriptors = [...(res.matchedDescriptors || []), matchDescriptor.descriptors]
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
        chosenDetections = templateDetections[i][0]
        break
      }
    }
    return null
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

/**
 * get max distance from image
 * @param {FaceMatcher} faceMatcher 
 * @param {url} image 
 */
const getMaxDistance = async (faceMatcher, image) => {
  const minDistance = 0.0
  const loadedImage = await canvas.loadImage(image)
  const resultsQuery = await faceapi.detectAllFaces(loadedImage, faceDetectionOptions)
    .withFaceLandmarks()
    .withFaceDescriptors()

  let maxDistance = 0
  for (let i = 0; i < resultsQuery.length; i++) {
    const bestMatch = faceMatcher.findBestMatch(resultsQuery[i].descriptor)
    if (bestMatch.label !== 'unknown' && bestMatch.distance > minDistance) {
      if (bestMatch.distance > maxDistance) {
        maxDistance = bestMatch.distance
      }
    }
  }
  return maxDistance
}
/**
 * get image that have largest distance
 * @param {FaceMatcher} faceMatcher 
 * @param {urls} images 
 */
const getMaxDistanceImage = async (faceMatcher, images) => {
  let maxDistance = 0
  let maxDistanceImageIndex = -1
  let appearCount = 0
  for (let i = 0; i < images.length; i++) {
    const distance = await getMaxDistance(faceMatcher, images[i])
    if (distance > 0) appearCount++
    if (distance > maxDistance) {
      maxDistance = distance
      maxDistanceImageIndex = i
    }
  }
  return {
    maxDistance,
    index: maxDistanceImageIndex,
    appearCount,
  }
}
const getMaxDistanceFriend = async (faceMatcher, friends) => {
  let maxDistance = 0
  let maxDistanceFriendIndex = -1
  for (let i = 0; i < friends.length; i++) {
    const distance = (await getMaxDistanceImage(faceMatcher, friends[i].images)).maxDistance
    if (distance > maxDistance) {
      maxDistance = distance
      maxDistanceFriendIndex = i
    }
  }
  return {
    maxDistance,
    index: maxDistanceFriendIndex,
  }
}
const getMaxAppearCountFriend = async (faceMatcher, friends) => {
  let maxAppearCount = 0
  let maxAppearCountFriendIndex = -1
  let maxDistance = 0

  try {
    await Promise.all(friends.map(async (friend, i) => {
      const maxDistanceImage = await getMaxDistanceImage(faceMatcher, friend.images)
      if (maxDistanceImage.appearCount > maxAppearCount) {
        maxAppearCount = maxDistanceImage.appearCount
        maxAppearCountFriendIndex = i
        maxDistance = maxDistanceImage.maxDistance
      }
    }))
  } catch (err) {
    console.error(err)
  }
  // for (let i = 0; i < friends.length; i++) {
  //   const maxDistanceImage = await getMaxDistanceImage(faceMatcher, friends[i].images)
  //   if (maxDistanceImage.appearCount > maxAppearCount) {
  //     maxAppearCount = maxDistanceImage.appearCount
  //     maxAppearCountFriendIndex = i
  //     maxDistance = maxDistanceImage.maxDistance
  //   }
  // }
  return {
    maxDistance,
    index: maxAppearCountFriendIndex,
  }
}

const getMaxAppearFriendIndex = async (faceMatcher, friends) => {
  let maxAppearance = 0
  let maxAppearFriendIndex = -1
  for (let i = 0; i < friends.length; i++) {
    console.log(`checking ${friends[i].name}`)
    const count = await checkFriendAppearance(faceMatcher, friends[i].images)

    if (count > maxAppearance) {
      maxAppearFriendIndex = i
      maxAppearance = count
    }
  }
  return maxAppearFriendIndex
}
exports.detect = async (req, res, next) => {
  const selectedDetection = await selectDetection(req.body.templateImages)
  const selectedDetectionDescriptors = [selectedDetection.descriptor]
  selectedDetection.matchedDescriptors.forEach(descriptors => {
    descriptors.forEach(descriptor => selectedDetectionDescriptors.push(descriptor))
  })
  const faceMatcher = new faceapi.FaceMatcher([new faceapi.LabeledFaceDescriptors(
    'person',
    // selectedDetections.map(d => d.descriptor)
    selectedDetectionDescriptors
  ),])
  const friends = req.body.listFriends
  // const maxAppearFriendIndex = await getMaxAppearFriendIndex(faceMatcher, friends)
  // if (maxAppearFriendIndex !== -1) return res.status(200).send(req.body.listFriends[maxAppearFriendIndex].name)

  const maxAppearCountFriend = await getMaxAppearCountFriend(faceMatcher, friends)
  if (maxAppearCountFriend.index !== -1) {
    return res.status(200).send(`${friends[maxAppearCountFriend.index].name}|${maxAppearCountFriend.maxDistance}`)
  }
  res.status(200).send({ message: 'No friend is valid' })
}

exports.checkPayload = (req, res, next) => {
  // check templateImages
  try {
    req.body.templateImages = req.body.templateImages.map(image => {
      try {
        const splitted = image.split('|')
        if (splitted.length !== 2) {
          throw new Error(`Invalid image: ${image}`)
        }
        return `${splitted[0]}${splitted[1]}`
      } catch (err) {
        throw err
      }
    })
  } catch (err) {
    console.error('checkPayload', err)
    return res.status(401).send({ message: err.message })
  }
  // check friends images
  try {
    req.body.listFriends.forEach(friend => {
      friend.images = friend.images.map(image => {
        try {
          const splitted = image.split('|')
          if (splitted.length !== 2) {
            throw new Error(`Invalid image: ${image}`)
          }
          return `${splitted[0]}${splitted[1]}`
        } catch (err) {
          throw err
        }
      })
    })
  } catch (err) {
    console.error('checkPayload', err)
    return res.status(401).send({ message: err.message })
  }
  next()
}