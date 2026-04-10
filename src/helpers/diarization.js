const fs = require("fs");
const fsPromises = require("fs").promises;
const path = require("path");
const { spawn } = require("child_process");
const debugLogger = require("./debugLogger");
const { downloadFile, createDownloadSignal, checkDiskSpace } = require("./downloadUtils");
const { resolveBinaryPath, gracefulStopProcess } = require("../utils/serverUtils");
const { getModelsDirForService } = require("./modelDirUtils");
const { convertToWav } = require("./ffmpegUtils");
const { getSafeTempDir } = require("./safeTempDir");

const DIARIZATION_TIMEOUT_MS = 300000; // 5 minutes

const SEGMENTATION_MODEL_URL =
  "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2";
const EMBEDDING_MODEL_URL =
  "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx";
const SILERO_VAD_MODEL_URL =
  "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx";

const SEGMENTATION_DIR = "sherpa-onnx-pyannote-segmentation-3-0";
const SEGMENTATION_ONNX = path.join(SEGMENTATION_DIR, "model.onnx");
const EMBEDDING_ONNX = "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx";
const SILERO_VAD_ONNX = "silero_vad.onnx";

class DiarizationManager {
  constructor() {
    this._process = null;
    this.currentDownloadProcess = null;
    this.cachedBinaryPath = null;
  }

  getBinaryPath() {
    if (this.cachedBinaryPath) return this.cachedBinaryPath;

    const platformArch = `${process.platform}-${process.arch}`;
    const binaryName =
      process.platform === "win32"
        ? `sherpa-onnx-diarize-${platformArch}.exe`
        : `sherpa-onnx-diarize-${platformArch}`;

    const resolved = resolveBinaryPath(binaryName);
    if (resolved) this.cachedBinaryPath = resolved;
    return resolved;
  }

  isAvailable() {
    return this.getBinaryPath() !== null && this.isModelDownloaded();
  }

  getModelsDir() {
    return getModelsDirForService("diarization");
  }

  getBundledModelsDir() {
    if (!process.resourcesPath) {
      return null;
    }

    return path.join(process.resourcesPath, "bin", "diarization-models");
  }

  _resolveModelPath(relativePath) {
    const bundledModelsDir = this.getBundledModelsDir();
    if (bundledModelsDir) {
      const bundledPath = path.join(bundledModelsDir, relativePath);
      if (fs.existsSync(bundledPath)) {
        return bundledPath;
      }
    }

    return path.join(this.getModelsDir(), relativePath);
  }

  isModelDownloaded() {
    const segPath = this._resolveModelPath(SEGMENTATION_ONNX);
    const embPath = this._resolveModelPath(EMBEDDING_ONNX);
    return fs.existsSync(segPath) && fs.existsSync(embPath);
  }

  getVadModelPath() {
    return this._resolveModelPath(SILERO_VAD_ONNX);
  }

  isVadModelDownloaded() {
    return fs.existsSync(this.getVadModelPath());
  }

  async downloadModels(progressCallback = null) {
    const modelsDir = this.getModelsDir();
    await fsPromises.mkdir(modelsDir, { recursive: true });

    const modelsReady = this.isModelDownloaded();
    const vadReady = this.isVadModelDownloaded();

    if (modelsReady && vadReady) {
      return { success: true, path: modelsDir };
    }

    const requiredBytes = modelsReady ? 2 * 1_000_000 : 37 * 1_000_000;
    const spaceCheck = await checkDiskSpace(modelsDir, requiredBytes * 2.5);
    if (!spaceCheck.ok) {
      throw new Error(
        `Not enough disk space. Need ~${Math.round((requiredBytes * 2.5) / 1_000_000)}MB, ` +
          `only ${Math.round(spaceCheck.availableBytes / 1_000_000)}MB available.`
      );
    }

    const { signal, abort } = createDownloadSignal();
    this.currentDownloadProcess = { abort };

    try {
      // Download segmentation model (tar.bz2)
      const segArchivePath = path.join(modelsDir, `${SEGMENTATION_DIR}.tar.bz2`);
      const segModelPath = path.join(modelsDir, SEGMENTATION_ONNX);

      if (!fs.existsSync(segModelPath)) {
        await downloadFile(SEGMENTATION_MODEL_URL, segArchivePath, {
          timeout: 600000,
          signal,
          onProgress: (downloadedBytes, totalBytes) => {
            if (progressCallback) {
              progressCallback({
                type: "progress",
                stage: "segmentation",
                downloaded_bytes: downloadedBytes,
                total_bytes: totalBytes,
                percentage: totalBytes > 0 ? Math.round((downloadedBytes / totalBytes) * 100) : 0,
              });
            }
          },
        });

        // Extract tar.bz2
        if (progressCallback) {
          progressCallback({ type: "progress", stage: "extracting", percentage: 100 });
        }

        await this._extractTarBz2(segArchivePath, modelsDir);
        await fsPromises.unlink(segArchivePath).catch(() => {});

        if (!fs.existsSync(segModelPath)) {
          throw new Error("Segmentation model extraction failed: model.onnx not found");
        }
      }

      // Download embedding model (.onnx directly)
      const embModelPath = path.join(modelsDir, EMBEDDING_ONNX);

      if (!fs.existsSync(embModelPath)) {
        await downloadFile(EMBEDDING_MODEL_URL, embModelPath, {
          timeout: 600000,
          signal,
          onProgress: (downloadedBytes, totalBytes) => {
            if (progressCallback) {
              progressCallback({
                type: "progress",
                stage: "embedding",
                downloaded_bytes: downloadedBytes,
                total_bytes: totalBytes,
                percentage: totalBytes > 0 ? Math.round((downloadedBytes / totalBytes) * 100) : 0,
              });
            }
          },
        });
      }

      if (!this.isVadModelDownloaded()) {
        try {
          await downloadFile(SILERO_VAD_MODEL_URL, this.getVadModelPath(), {
            timeout: 600000,
            signal,
            onProgress: (downloadedBytes, totalBytes) => {
              if (progressCallback) {
                progressCallback({
                  type: "progress",
                  stage: "vad",
                  downloaded_bytes: downloadedBytes,
                  total_bytes: totalBytes,
                  percentage: totalBytes > 0 ? Math.round((downloadedBytes / totalBytes) * 100) : 0,
                });
              }
            },
          });
        } catch (error) {
          if (error.isAbort) {
            throw new Error("Download interrupted by user");
          }
          debugLogger.warn("Silero VAD model download failed", {
            error: error.message,
            modelsDir,
          });
        }
      }

      if (progressCallback) {
        progressCallback({ type: "complete", percentage: 100 });
      }

      debugLogger.info("Diarization models downloaded", { modelsDir });
      return { success: true, path: modelsDir };
    } catch (error) {
      if (error.isAbort) {
        throw new Error("Download interrupted by user");
      }
      if (progressCallback) {
        progressCallback({ type: "error", error: error.message });
      }
      throw error;
    } finally {
      this.currentDownloadProcess = null;
    }
  }

  async _extractTarBz2(archivePath, destDir) {
    try {
      await this._runSystemTar(archivePath, destDir);
      return;
    } catch (err) {
      debugLogger.debug("System tar failed, falling back to JS extraction", {
        error: err.message,
      });
    }

    const unbzip2 = require("unbzip2-stream");
    const tar = require("tar");
    const { pipeline } = require("stream/promises");
    await pipeline(fs.createReadStream(archivePath), unbzip2(), tar.x({ cwd: destDir }));
  }

  _runSystemTar(archivePath, destDir) {
    return new Promise((resolve, reject) => {
      // Use relative paths from archive dir as cwd so neither -f nor -C args
      // contain Windows drive letter colons (GNU tar treats C: as remote host)
      const cwd = path.dirname(archivePath);
      const tarProcess = spawn(
        "tar",
        ["-xjf", path.basename(archivePath), "-C", path.relative(cwd, destDir)],
        { stdio: ["ignore", "pipe", "pipe"], cwd }
      );

      let stderr = "";

      tarProcess.stderr.on("data", (data) => {
        stderr += data.toString();
      });

      tarProcess.on("close", (code) => {
        if (code === 0) {
          resolve();
        } else {
          reject(new Error(`tar extraction failed with code ${code}: ${stderr}`));
        }
      });

      tarProcess.on("error", (err) => {
        reject(new Error(`Failed to start tar process: ${err.message}`));
      });
    });
  }

  async cancelDownload() {
    if (this.currentDownloadProcess) {
      this.currentDownloadProcess.abort();
      this.currentDownloadProcess = null;
      return { success: true, message: "Download cancelled" };
    }
    return { success: false, error: "No active download to cancel" };
  }

  async diarize(wavPath, options = {}) {
    const { numSpeakers = -1, threshold = 0.5 } = options;

    const binaryPath = this.getBinaryPath();
    if (!binaryPath) {
      debugLogger.warn("Diarization binary not found");
      return [];
    }

    if (!this.isModelDownloaded()) {
      debugLogger.warn("Diarization models not downloaded");
      return [];
    }

    if (!fs.existsSync(wavPath)) {
      debugLogger.warn("Diarization input file not found", { wavPath });
      return [];
    }

    const segPath = this._resolveModelPath(SEGMENTATION_ONNX);
    const embPath = this._resolveModelPath(EMBEDDING_ONNX);

    const args = [
      `--segmentation.pyannote-model=${segPath}`,
      `--embedding.model=${embPath}`,
      `--clustering.num-clusters=${numSpeakers}`,
      `--clustering.cluster-threshold=${threshold}`,
      "--min-duration-on=0.2",
      "--min-duration-off=0.5",
      wavPath,
    ];

    debugLogger.info("Starting diarization", {
      binaryPath,
      numSpeakers,
      threshold,
      wavPath,
    });

    return new Promise((resolve) => {
      let stdout = "";
      let stderr = "";

      const proc = spawn(binaryPath, args, {
        stdio: ["ignore", "pipe", "pipe"],
        windowsHide: true,
      });

      this._process = proc;

      const timeout = setTimeout(() => {
        debugLogger.warn("Diarization timed out", { timeoutMs: DIARIZATION_TIMEOUT_MS });
        gracefulStopProcess(proc);
        this._process = null;
        resolve([]);
      }, DIARIZATION_TIMEOUT_MS);

      proc.stdout.on("data", (data) => {
        stdout += data.toString();
      });

      proc.stderr.on("data", (data) => {
        stderr += data.toString();
      });

      proc.on("close", (code) => {
        clearTimeout(timeout);
        this._process = null;

        if (code !== 0) {
          debugLogger.warn("Diarization process exited with error", {
            code,
            stderr: stderr.slice(-500).trim(),
          });
          resolve([]);
          return;
        }

        const segments = this._parseOutput(stdout);
        debugLogger.info("Diarization complete", { segmentCount: segments.length });
        resolve(segments);
      });

      proc.on("error", (err) => {
        clearTimeout(timeout);
        this._process = null;
        debugLogger.warn("Diarization process error", { error: err.message });
        resolve([]);
      });
    });
  }

  _parseOutput(stdout) {
    const segments = [];
    const lineRegex = /^(\d+\.?\d*)\s+--\s+(\d+\.?\d*)\s+(speaker_\d+)$/;

    for (const line of stdout.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      const match = trimmed.match(lineRegex);
      if (match) {
        segments.push({
          start: parseFloat(match[1]),
          end: parseFloat(match[2]),
          speaker: match[3],
        });
      }
    }

    return segments;
  }

  mergeWithTranscript(transcriptSegments, diarizationSegments) {
    if (!transcriptSegments || transcriptSegments.length === 0) return [];
    if (!diarizationSegments || diarizationSegments.length === 0) {
      return transcriptSegments.map((seg) => ({ ...seg }));
    }

    // Build speaker renumbering map (e.g., speaker_00 → speaker_0)
    const speakerSet = new Set(diarizationSegments.map((d) => d.speaker));
    const speakerMap = new Map();
    let idx = 0;
    for (const sp of speakerSet) {
      speakerMap.set(sp, `speaker_${idx}`);
      idx++;
    }

    return transcriptSegments.map((seg) => {
      const enriched = { ...seg };

      if (seg.source === "mic") {
        enriched.speaker = "you";
        return enriched;
      }

      if (seg.source === "system" && seg.timestamp != null) {
        const ts = seg.timestamp;
        let bestSpeaker = null;
        let bestOverlap = 0;

        for (const dSeg of diarizationSegments) {
          if (ts >= dSeg.start && ts <= dSeg.end) {
            const overlap = dSeg.end - dSeg.start;
            if (overlap > bestOverlap) {
              bestOverlap = overlap;
              bestSpeaker = dSeg.speaker;
            }
          }
        }

        if (bestSpeaker) {
          enriched.speaker = speakerMap.get(bestSpeaker) || bestSpeaker;
        }
      }

      return enriched;
    });
  }

  async convertRawPcmToWav(rawPcmPath, inputSampleRate) {
    const stat = await fsPromises.stat(rawPcmPath);
    if (stat.size === 0) {
      throw new Error("Raw PCM file is empty");
    }

    const tempDir = getSafeTempDir();
    const timestamp = Date.now();
    const inputWavPath = path.join(tempDir, `ow-diarize-${timestamp}-input.wav`);
    const wavPath = path.join(tempDir, `ow-diarize-${timestamp}.wav`);

    // Stream: write 44-byte WAV header, then pipe raw PCM — avoids loading entire file into memory
    const header = this._createWavHeader(stat.size, inputSampleRate, 1);
    await new Promise((resolve, reject) => {
      const out = fs.createWriteStream(inputWavPath);
      out.write(header);
      const pcmStream = fs.createReadStream(rawPcmPath);
      pcmStream.pipe(out);
      out.on("finish", resolve);
      out.on("error", reject);
      pcmStream.on("error", reject);
    });

    try {
      await convertToWav(inputWavPath, wavPath, { sampleRate: 16000, channels: 1 });
    } finally {
      await fsPromises.unlink(inputWavPath).catch(() => {});
    }

    debugLogger.debug("Raw PCM converted to WAV for diarization", {
      wavPath,
      rawPcmBytes: stat.size,
    });

    return wavPath;
  }

  _createWavHeader(dataSize, sampleRate, channels) {
    const bytesPerSample = 2;
    const blockAlign = channels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const header = Buffer.alloc(44);

    header.write("RIFF", 0);
    header.writeUInt32LE(36 + dataSize, 4);
    header.write("WAVE", 8);
    header.write("fmt ", 12);
    header.writeUInt32LE(16, 16);
    header.writeUInt16LE(1, 20);
    header.writeUInt16LE(channels, 22);
    header.writeUInt32LE(sampleRate, 24);
    header.writeUInt32LE(byteRate, 28);
    header.writeUInt16LE(blockAlign, 32);
    header.writeUInt16LE(bytesPerSample * 8, 34);
    header.write("data", 36);
    header.writeUInt32LE(dataSize, 40);

    return header;
  }

  async deleteModels() {
    const modelsDir = this.getModelsDir();
    const segDir = path.join(modelsDir, SEGMENTATION_DIR);
    const embPath = path.join(modelsDir, EMBEDDING_ONNX);
    const vadPath = this.getVadModelPath();

    if (fs.existsSync(segDir)) {
      await fsPromises.rm(segDir, { recursive: true, force: true });
    }
    if (fs.existsSync(embPath)) {
      await fsPromises.unlink(embPath);
    }
    if (fs.existsSync(vadPath)) {
      await fsPromises.unlink(vadPath);
    }

    debugLogger.info("Diarization models deleted", { modelsDir });
    return { success: true };
  }

  async shutdown() {
    if (this._process) {
      await gracefulStopProcess(this._process);
      this._process = null;
    }
  }
}

module.exports = DiarizationManager;
