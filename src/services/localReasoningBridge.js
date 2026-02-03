const modelManager = require("../helpers/modelManagerBridge").default;
const debugLogger = require("../helpers/debugLogger");
const { getSystemPrompt } = require("../helpers/prompts");

class LocalReasoningService {
  constructor() {
    this.isProcessing = false;
  }

  async isAvailable() {
    try {
      // Check if llama.cpp is installed
      await modelManager.ensureLlamaCpp();

      // Check if at least one model is downloaded
      const models = await modelManager.getAllModels();
      return models.some((model) => model.isDownloaded);
    } catch (error) {
      return false;
    }
  }

  async processText(text, modelId, agentName = null, config = {}) {
    debugLogger.logReasoning("LOCAL_BRIDGE_START", {
      modelId,
      agentName,
      textLength: text.length,
      hasConfig: Object.keys(config).length > 0,
    });

    if (process.env.REASONING_PROVIDER && process.env.REASONING_PROVIDER !== "local") {
      throw new Error(
        `Local reasoning is disabled while provider is set to ${process.env.REASONING_PROVIDER}`
      );
    }

    if (this.isProcessing) {
      throw new Error("Already processing a request");
    }

    this.isProcessing = true;
    const startTime = Date.now();

    try {
      const models = await modelManager.getAllModels();
      const matchedModel = models.find((model) => model.id === modelId);
      if (!matchedModel || !matchedModel.isDownloaded) {
        throw new Error(`Local model ${modelId} is not downloaded`);
      }

      debugLogger.logReasoning("LOCAL_BRIDGE_PROMPT", {
        promptLength: text.length,
        hasAgentName: !!agentName,
      });

      const inferenceConfig = {
        maxTokens: config.maxTokens || this.calculateMaxTokens(text.length),
        temperature: config.temperature || 0.7,
        topK: config.topK || 40,
        topP: config.topP || 0.9,
        repeatPenalty: config.repeatPenalty || 1.1,
        contextSize: config.contextSize || 4096,
        threads: config.threads || 4,
        systemPrompt: getSystemPrompt(agentName, config.customDictionary),
      };

      debugLogger.logReasoning("LOCAL_BRIDGE_INFERENCE", {
        modelId,
        config: inferenceConfig,
      });

      // Run inference
      const result = await modelManager.runInference(modelId, text, inferenceConfig);

      const processingTime = Date.now() - startTime;

      debugLogger.logReasoning("LOCAL_BRIDGE_SUCCESS", {
        modelId,
        processingTimeMs: processingTime,
        resultLength: result.length,
        resultPreview: result.substring(0, 100) + (result.length > 100 ? "..." : ""),
      });

      return result;
    } catch (error) {
      const processingTime = Date.now() - startTime;

      debugLogger.logReasoning("LOCAL_BRIDGE_ERROR", {
        modelId,
        processingTimeMs: processingTime,
        error: error.message,
        stack: error.stack,
      });

      throw error;
    } finally {
      this.isProcessing = false;
    }
  }

  calculateMaxTokens(textLength, minTokens = 100, maxTokens = 2048, multiplier = 2) {
    return Math.max(minTokens, Math.min(textLength * multiplier, maxTokens));
  }
}

module.exports = {
  default: new LocalReasoningService(),
};
