const mongoose = require("mongoose");

const jobSchema = new mongoose.Schema({
    filename: String,
    status: { type: String, default: "pending" },  // pending | processing | done | failed
    createdAt: { type: Date, default: Date.now }
});

module.exports = mongoose.model("Job", jobSchema);
