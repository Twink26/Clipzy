const express = require("express");
const multer = require("multer");
const Job = require("../models/Job");

const router = express.Router();

// File storage setup
const storage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, "uploads/"),
    filename: (req, file, cb) => cb(null, Date.now() + "-" + file.originalname),
});
const upload = multer({ storage });

// Upload endpoint
router.post("/", upload.single("video"), async (req, res) => {
    try {
        // Save job in DB
        const job = new Job({
            filename: req.file.filename,
            status: "pending",
        });

        await job.save();

        res.json({ message: "Upload successful", jobId: job._id });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

module.exports = router;
