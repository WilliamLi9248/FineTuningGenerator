# Fine-Tuning Training Data Generator

A comprehensive Python tool for generating training datasets for various Large Language Model (LLM) fine-tuning platforms. This utility supports multiple commercial and open-source providers, offering flexible content generation options and standardized output formats.

## Overview

The Fine-Tuning Training Data Generator creates structured conversation datasets tailored to specific use cases. It supports nine different output formats and six content generation methods, making it suitable for both research and production environments.

## Supported Platforms

### Commercial Cloud Providers
- **OpenAI** - GPT-3.5/4 fine-tuning with messages format
- **Claude (Amazon Bedrock)** - Claude 3 fine-tuning with system/messages structure
- **Gemini (Vertex AI)** - Google Cloud fine-tuning with input/output pairs
- **Hugging Face** - Open-source model fine-tuning with flexible formatting

### Local and Open Source Models
- **Llama** - Meta's Llama 2/3 with chat template format
- **Alpaca** - Stanford Alpaca instruction-following format
- **ShareGPT** - Conversational dataset format for multi-turn dialogue
- **DeepSeek** - Reasoning-enhanced format for DeepSeek models
- **Unsloth** - Memory-optimized format for efficient training

## Content Generation Options

The tool provides multiple methods for generating training content:

1. **Demo Mode** - Pre-built expert responses with high consistency
2. **Ollama** - Local LLM instance for unlimited free generation
3. **OpenAI API** - GPT-3.5-turbo for professional-quality content
4. **Claude API** - Anthropic's Claude 3 Haiku for thoughtful responses
5. **Hugging Face API** - Free tier access to various models
6. **Groq API** - High-speed inference for rapid dataset creation

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

For local generation with Ollama:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download a model
ollama pull llama3.1:8b

# Start the service
ollama serve
```

## Usage

### Interactive Mode

Run the tool in interactive mode for guided setup:

```bash
python fine_tuning_generator.py --interactive
```

This mode provides step-by-step configuration including:
- Target platform selection
- Task description input
- Number of examples specification
- Content generation method choice
- Output file naming

### Command Line Interface

Generate datasets directly from the command line:

```bash
python fine_tuning_generator.py --provider openai --task "customer service chatbot" --examples 100 --llm demo
```

### API Key Configuration

For commercial APIs, set environment variables:

```bash
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_claude_key"
export GROQ_API_KEY="your_groq_key"
export HUGGINGFACE_API_KEY="your_hf_token"
```

## Command Line Arguments

- `--provider` - Target platform (openai, claude, gemini, huggingface, llama, alpaca, sharegpt, deepseek, unsloth)
- `--task` - Description of the fine-tuning objective
- `--examples` - Number of training examples to generate (default: 10)
- `--output` - Custom output filename (optional)
- `--llm` - Content generation method (demo, ollama, openai, claude, huggingface, groq)
- `--interactive` - Launch interactive configuration mode

## Output Format Examples

### OpenAI Format
```json
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "How can I help you today?"}, {"role": "assistant", "content": "I'm here to assist with any questions you have."}]}
```

### Alpaca Format
```json
{"instruction": "Solve this equation: 2x + 5 = 15", "input": "", "output": "To solve 2x + 5 = 15, subtract 5 from both sides to get 2x = 10, then divide by 2 to get x = 5."}
```

### ShareGPT Format
```json
{"id": "sharegpt_001", "conversations": [{"from": "human", "value": "Explain quantum computing"}, {"from": "gpt", "value": "Quantum computing uses quantum mechanical phenomena..."}]}
```

## Use Cases

### Customer Service Training
Generate realistic customer support conversations with varied scenarios including order issues, returns, account problems, and billing inquiries.

### Code Review Assistant
Create datasets for programming assistance with code optimization, debugging, security reviews, and best practices guidance.

### Educational Content
Develop training data for tutoring systems covering mathematics, programming concepts, and technical explanations.

### Specialized Domains
Generate domain-specific conversations for robotics, healthcare, finance, or other specialized fields requiring expert knowledge.

## Quality Considerations

- **Context Awareness** - Responses are generated based on task-specific context and domain knowledge
- **Variability** - Multiple system prompt variations and response patterns prevent repetitive training data
- **Relevance** - Content matches the specified use case with appropriate technical depth
- **Consistency** - Maintains professional tone and accurate information across examples

## Technical Requirements

- Python 3.7 or higher
- Internet connection for API-based generation
- Optional: Local Ollama installation for free unlimited generation
- Optional: API keys for commercial services

## File Formats

All outputs are generated in JSONL (JSON Lines) format, with each line containing a complete training example. This format is widely supported across fine-tuning platforms and can be easily processed by training pipelines.

## Performance Guidelines

- **Demo Mode** - Instant generation, suitable for rapid prototyping
- **Ollama** - Local processing, no API limits, requires adequate hardware
- **Commercial APIs** - High quality but usage costs, rate limiting may apply
- **Batch Processing** - Recommended for large datasets (100+ examples)

## Error Handling

The tool includes comprehensive error handling with automatic fallback to demo mode when:
- API keys are missing or invalid
- Network connectivity issues occur
- Service rate limits are exceeded
- Local services (Ollama) are unavailable

## License

This project is released under the MIT License, allowing free use and modification for both personal and commercial applications.

## Support

For technical issues, feature requests, or contributions, please refer to the project documentation or contact the development team.