const router = require('express').Router()
const detectionController = require('../controllers/detection')

router.post('/', detectionController.detect);

module.exports = router;
