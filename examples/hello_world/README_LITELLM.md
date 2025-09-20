# Hello World with Local LLMs (LiteLLM + Ollama)

<p>
  <a href="https://colab.research.google.com/github/typedef-ai/fenic/blob/main/examples/hello_world/hello_world_litellm.ipynb">
    <img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
  </a>
</p>

This example demonstrates how to use Fenic with **local LLMs via LiteLLM and Ollama** instead of cloud APIs. Perfect for privacy-sensitive workflows, cost control, or offline development!

## What You'll Learn

- How to configure Fenic with local models through LiteLLM
- Error log analysis without sending data to external APIs
- Zero-cost semantic operations using local inference
- Benefits of local vs cloud LLM processing

## Prerequisites

### 1. Install Ollama

Download and install Ollama from [ollama.ai](https://ollama.ai)

### 2. Pull Required Models

```bash
# Pull the recommended model (30B parameters, great for analysis)
ollama pull qwen3:30b

# Or try alternatives:
ollama pull gpt-oss        # Open source GPT-style model
ollama pull deepseek-r1    # Reasoning-focused model
```

### 3. Verify Ollama is Running

```bash
# Check that Ollama is running
curl http://localhost:11434/api/version
```

### 4. Install Dependencies

```bash
pip install fenic litellm matplotlib seaborn polars==1.30.0
```

## Key Differences from Cloud APIs

| Aspect | Cloud APIs (OpenAI, etc.) | Local LLMs (LiteLLM + Ollama) |
|--------|---------------------------|--------------------------------|
| **Privacy** | Data sent to external servers | All processing local |
| **Cost** | Per-token pricing | Hardware cost only |
| **Latency** | Network + processing | Processing only |
| **Availability** | Depends on service | Always available |
| **Rate Limits** | API provider limits | Hardware limits only |
| **Internet** | Required | Not required |

## Available Models in This Example

The notebook is configured to work with three models (all with 16k context windows):

1. **`qwen3:30b`** - Qwen3 30B model, excellent for complex analysis
2. **`gpt-oss`** - Open source GPT-style model
3. **`deepseek-r1`** - Reasoning-focused model for complex logic

## Example Output

```
🤖 Starting local LLM analysis...
⏳ This may take a moment with local models...

Analyzing log 1/3: api-gateway...
  ✅ Severity: high
  📋 Root Cause: NullPointerException due to missing user data...

📊 LOCAL LLM PROCESSING METRICS
==================================================
🔢 Total Requests: 3
📝 Input Tokens: 1,247
📤 Output Tokens: 342
💰 Total Cost: $0.00 (FREE!)
🏠 Model: qwen3:30b (Local)
🔗 Endpoint: http://localhost:11434

✅ All processing completed locally - no data sent to external APIs!
```

## Use Cases Perfect for Local LLMs

### 🔒 **Privacy-Sensitive Environments**
- Healthcare records analysis
- Financial data processing
- Legal document review
- Personal data analysis

### 🏢 **Enterprise & Regulated Industries**
- Air-gapped networks
- Compliance requirements
- Data sovereignty needs
- Security-first organizations

### 💰 **Cost-Conscious Applications**
- High-volume processing
- Experimentation and development
- Educational projects
- Prototype development

### 🚀 **Performance-Critical Applications**
- Real-time analysis
- Offline deployments
- Edge computing
- Consistent latency requirements

## Model Recommendations

| Model | Best For | Size | Performance |
|-------|----------|------|-------------|
| `qwen3:30b` | Complex analysis, reasoning | ~17GB | High quality |
| `gpt-oss` | General purpose, balanced | ~7GB | Good quality |
| `deepseek-r1` | Logic, code analysis | ~14GB | High reasoning |

## Performance Tips

1. **Hardware Requirements**:
   - GPU: 16GB+ VRAM for 30B models
   - CPU: 32GB+ RAM for CPU inference
   - Storage: 20GB+ free space

2. **Optimization**:
   - Use GPU acceleration when available
   - Batch similar requests together
   - Cache frequent analyses
   - Consider smaller models for development

3. **Model Selection**:
   - Start with smaller models for testing
   - Use 30B models for production quality
   - Fine-tune for domain-specific tasks

## Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
ollama list

# Restart Ollama service
ollama serve
```

### Model Not Found
```bash
# Pull the model
ollama pull qwen3:30b

# List available models
ollama list
```

### Memory Issues
```bash
# Use smaller models
ollama pull llama3:8b

# Or configure CPU-only inference
OLLAMA_GPU_LAYERS=0 ollama serve
```

## Next Steps

1. **Try Different Models**: Experiment with various Ollama models
2. **Scale Up**: Process larger datasets without API costs
3. **Customize**: Fine-tune models for your specific use cases
4. **Deploy**: Set up production pipelines with local inference
5. **Integrate**: Combine with existing workflows and systems

---

**Benefits Summary:**
- ✅ Complete privacy and data control
- ✅ Zero ongoing inference costs
- ✅ No internet dependency
- ✅ Consistent performance
- ✅ Unlimited usage
- ✅ Full customization control