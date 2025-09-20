# Fenic LiteLLM Examples - Local AI Processing

This collection demonstrates how to use Fenic with local language models via LiteLLM and Ollama, enabling **complete privacy**, **zero API costs**, and **offline AI processing**.

## Overview

LiteLLM integration with Fenic allows you to:
- 🔒 **Complete Privacy**: All processing happens locally, no data leaves your machine
- 💰 **Zero API Costs**: Use powerful local models without any cloud API fees
- 🌐 **Offline Processing**: Work with AI capabilities without internet connectivity
- 🚀 **High Performance**: Direct integration with local model servers like Ollama

## Prerequisites

### 1. Install Ollama
```bash
# Download and install from https://ollama.ai
# Or using Homebrew:
brew install ollama
```

### 2. Pull Required Models
```bash
# Pull the models used in these examples
ollama pull qwen3:30b      # Primary model for most examples
ollama pull gpt-oss        # Alternative model option
ollama pull deepseek-r1    # Another alternative option
```

### 3. Start Ollama Server
```bash
ollama serve
# Server will start on http://localhost:11434 by default
```

### 4. Install Fenic with LiteLLM Dependencies
```bash
pip install fenic
pip install litellm
```

## Examples Overview

| Example | Description | Key Features | Estimated Runtime |
|---------|-------------|--------------|-------------------|
| [News Analysis](news_analysis/news_analysis_litellm.py) | Analyze news articles for bias and sentiment | Classification, bias detection, media profiling | 2-3 minutes |
| [Meeting Transcripts](meeting_transcript_processing/transcript_processing_litellm.py) | Extract action items and insights from meeting transcripts | Structured extraction, transcript parsing | 3-4 minutes |
| [Semantic Joins](semantic_joins/semantic_joins_litellm.py) | Join datasets using semantic similarity | Similarity matching, data integration | 1-2 minutes |
| [Markdown Processing](markdown_processing/markdown_processing_litellm.py) | Process academic papers and documents | Document analysis, content extraction | 2-3 minutes |
| [Named Entity Recognition](named_entity_recognition/ner_litellm.py) | Extract security vulnerabilities from reports | Multi-stage NER, risk assessment | 4-5 minutes |

## Configuration

All examples use the same basic configuration pattern:

```python
import fenic as fc

# Create session config with LiteLLM
config = fc.SessionConfig(
    app_name="your_app_name",
    semantic=fc.SemanticConfig(
        language_models={
            "local_qwen": fc.LiteLLMLanguageModel(
                model_name="qwen3:30b",
                rpm=100,  # Requests per minute
                tpm=10000,  # Tokens per minute
                api_base="http://localhost:11434"
            )
        },
        default_language_model="local_qwen"
    )
)

# Create session
session = fc.Session.get_or_create(config)
```

## Running Examples

Each example is self-contained and includes sample data. Simply run:

```bash
cd examples/news_analysis
python news_analysis_litellm.py
```

## Performance Tips

### Model Selection
- **qwen3:30b**: Best overall performance, requires 16GB+ RAM
- **gpt-oss**: Lighter weight option, good for basic tasks
- **deepseek-r1**: Good for reasoning tasks

### Optimization
- Adjust `rpm` and `tpm` limits based on your hardware
- Use SSD storage for faster model loading
- Consider GPU acceleration if available with Ollama

### Hardware Requirements
| Model | Min RAM | Recommended RAM | Storage |
|-------|---------|-----------------|---------|
| qwen3:30b | 16GB | 32GB | ~17GB |
| gpt-oss | 8GB | 16GB | ~4GB |
| deepseek-r1 | 8GB | 16GB | ~4GB |

## Troubleshooting

### Common Issues

1. **"Model not found" error**
   ```bash
   # Pull the model first
   ollama pull qwen3:30b
   ```

2. **Connection refused**
   ```bash
   # Make sure Ollama is running
   ollama serve
   ```

3. **Out of memory errors**
   - Try a smaller model like `gpt-oss`
   - Reduce batch sizes in the examples
   - Close other applications to free memory

4. **Slow performance**
   - Check if your model is running on CPU vs GPU
   - Reduce max_output_tokens in configuration
   - Use faster models for development/testing

### Debugging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Advanced Configuration

### Custom Models
Add your own Ollama models to the catalog:
```python
# Pull your custom model
# ollama pull your-model:tag

# Use in configuration
fc.LiteLLMLanguageModel(
    model_name="your-model:tag",
    rpm=50,
    tpm=5000,
    api_base="http://localhost:11434"
)
```

### Rate Limiting
Adjust based on your hardware:
```python
# For powerful hardware
rpm=200, tpm=20000

# For limited resources
rpm=50, tpm=5000
```

### Multiple Models
Configure multiple models for different tasks:
```python
language_models={
    "fast": fc.LiteLLMLanguageModel(model_name="gpt-oss", rpm=100, tpm=10000),
    "powerful": fc.LiteLLMLanguageModel(model_name="qwen3:30b", rpm=50, tpm=5000),
}
```

## Privacy & Security

These examples ensure complete privacy by:
- Processing all data locally on your machine
- Never sending data to external APIs
- Using local model inference only
- No network dependencies after model download

## Cost Comparison

| Processing | Cloud API Cost | Local LiteLLM Cost |
|------------|---------------|-------------------|
| 1M tokens | $1-20 | $0 (electricity only) |
| News analysis example | ~$0.50 | $0 |
| Full NER pipeline | ~$2-5 | $0 |
| **Annual usage** | **$100-1000+** | **$0** |

## Contributing

To add new LiteLLM examples:
1. Follow the configuration pattern above
2. Include sample data and clear documentation
3. Test with multiple model sizes
4. Add performance estimates to this README

## Support

- **Fenic Issues**: [GitHub Issues](https://github.com/anthropics/fenic/issues)
- **LiteLLM Docs**: [LiteLLM Documentation](https://docs.litellm.ai/)
- **Ollama Support**: [Ollama GitHub](https://github.com/ollama/ollama)

---

*These examples demonstrate the power of local AI processing with Fenic, providing enterprise-grade data processing capabilities while maintaining complete control over your data and infrastructure.*