# LLM Provider Options Guide

The Fine-Tuning Generator supports multiple LLM providers for content generation. Choose the option that best fits your needs:

## Provider Options

### 1. Demo Mode (Pre-built Responses)
- **Cost**: Free
- **Setup**: None required
- **Quality**: High, expert-crafted responses
- **Speed**: Instant
- **Variety**: Limited but consistent
- **Best for**: Quick testing, consistent quality

```bash
python fine_tuning_generator.py --provider openai --task "customer service" --llm demo --examples 10
```

### 2. Ollama (Free Local LLM)
- **Cost**: Free
- **Setup**: Install Ollama locally
- **Quality**: Very high, creative responses
- **Speed**: Fast (local processing)
- **Variety**: High, AI-generated diversity
- **Best for**: Privacy-conscious users, unlimited generation

**Setup Instructions:**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download a model
ollama pull llama3.1:8b

# Start Ollama service
ollama serve
```

**Usage:**
```bash
python fine_tuning_generator.py --provider alpaca --task "robotics assistant" --llm ollama --examples 20
```

### 3. OpenAI API (Commercial)
- **Cost**: Pay per token (~$0.002/1k tokens)
- **Setup**: API key required
- **Quality**: Excellent, GPT-3.5-turbo
- **Speed**: Fast
- **Variety**: High, professional quality
- **Best for**: Production use, high-quality datasets

**Setup:**
```bash
export OPENAI_API_KEY="your_api_key_here"
```

**Usage:**
```bash
python fine_tuning_generator.py --provider sharegpt --task "coding assistant" --llm openai --examples 50
```

### 4. Claude API (Commercial)
- **Cost**: Pay per token (~$0.25/1M tokens)
- **Setup**: Anthropic API key required
- **Quality**: Excellent, Claude 3 Haiku
- **Speed**: Fast
- **Variety**: High, thoughtful responses
- **Best for**: Complex reasoning tasks, safety-conscious generation

**Setup:**
```bash
export ANTHROPIC_API_KEY="your_api_key_here"
```

**Usage:**
```bash
python fine_tuning_generator.py --provider deepseek --task "reasoning assistant" --llm claude --examples 30
```

### 5. Hugging Face (Free Tier)
- **Cost**: Free (with rate limits)
- **Setup**: HF token required
- **Quality**: Good, varies by model
- **Speed**: Moderate (API dependent)
- **Variety**: Good
- **Best for**: Experimentation, budget-conscious users

**Setup:**
```bash
export HUGGINGFACE_API_KEY="your_hf_token_here"
```

**Usage:**
```bash
python fine_tuning_generator.py --provider llama --task "chat assistant" --llm huggingface --examples 15
```

### 6. Groq API (Commercial)
- **Cost**: Pay per token (very fast inference)
- **Setup**: Groq API key required
- **Quality**: Excellent, Llama3-8B
- **Speed**: Very fast
- **Variety**: High
- **Best for**: Speed-critical applications, bulk generation

**Setup:**
```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

**Usage:**
```bash
python fine_tuning_generator.py --provider unsloth --task "programming tutor" --llm groq --examples 100
```

## Format Compatibility

All LLM providers work with all output formats:

| LLM Provider | OpenAI | Claude | Gemini | HuggingFace | Alpaca | ShareGPT | DeepSeek | Unsloth |
|--------------|---------|---------|---------|-------------|---------|-----------|-----------|----------|
| Demo         | Yes    | Yes    | Yes    | Yes        | Yes    | Yes      | Yes      | Yes     |
| Ollama       | Yes    | Yes    | Yes    | Yes        | Yes    | Yes      | Yes      | Yes     |
| OpenAI       | Yes    | Yes    | Yes    | Yes        | Yes    | Yes      | Yes      | Yes     |
| Claude       | Yes    | Yes    | Yes    | Yes        | Yes    | Yes      | Yes      | Yes     |
| HuggingFace  | Yes    | Yes    | Yes    | Yes        | Yes    | Yes      | Yes      | Yes     |
| Groq         | Yes    | Yes    | Yes    | Yes        | Yes    | Yes      | Yes      | Yes     |

## Recommendations

### For Different Use Cases:

**Quick Testing & Prototyping:**
- Use `--llm demo` for instant, consistent results

**Research & Development:**
- Use `--llm ollama` for free, unlimited, high-quality generation

**Production Datasets:**
- Use `--llm openai` or `--llm claude` for premium quality

**Budget-Conscious:**
- Use `--llm huggingface` for free API access
- Use `--llm ollama` for unlimited free local generation

**Speed-Critical:**
- Use `--llm groq` for fastest API responses
- Use `--llm ollama` for fast local processing

### Sample Commands:

```bash
# Quick demo with pre-built responses
python fine_tuning_generator.py --provider openai --task "customer support" --llm demo --examples 10

# High-quality local generation
python fine_tuning_generator.py --provider alpaca --task "robotics collaboration" --llm ollama --examples 50

# Production-quality with OpenAI
python fine_tuning_generator.py --provider sharegpt --task "coding mentor" --llm openai --examples 100

# Interactive mode with all options
python fine_tuning_generator.py --interactive
```

## Troubleshooting

### Common Issues:

1. **"API key not found"** - Set environment variables for commercial APIs
2. **"Ollama server not running"** - Start Ollama with `ollama serve`
3. **"Model not found"** - Download models with `ollama pull model_name`
4. **Rate limits** - Use `--llm demo` or `--llm ollama` for unlimited generation

### Fallback Behavior:

If any LLM provider fails, the system automatically falls back to demo mode to ensure your data generation never fails completely.

## Quality Comparison

| Provider | Technical Accuracy | Creativity | Consistency | Cost | Speed |
|----------|-------------------|------------|-------------|------|-------|
| Demo     | Excellent         | Fair       | Excellent   | Free | Excellent |
| Ollama   | Excellent         | Excellent  | Good        | Free | Good      |
| OpenAI   | Excellent         | Excellent  | Excellent   | $$   | Good      |
| Claude   | Excellent         | Excellent  | Excellent   | $$   | Good      |
| HF       | Good              | Good       | Good        | Free | Fair      |
| Groq     | Excellent         | Excellent  | Good        | $    | Excellent |