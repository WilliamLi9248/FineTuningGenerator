# Fine-Tuning Training Data Generator

A Python tool that generates training files for various LLM fine-tuning platforms based on user requirements. Supports OpenAI, Claude (Bedrock), Gemini, and Hugging Face formats.

## Features

- **Multi-Provider Support**: Generate training data for OpenAI, Claude, Gemini, and Hugging Face
- **Format-Specific Output**: Automatically formats data according to each provider's requirements
- **LLM Integration**: Uses various LLM providers (local Ollama, Hugging Face, demo mode) to generate content
- **Interactive Mode**: User-friendly interface for easy configuration
- **Command Line Interface**: Batch processing support for automation

## Supported Formats

### Commercial/Cloud Providers

#### OpenAI
- **Format**: JSONL
- **Structure**: `{"messages": [{"role": "system/user/assistant", "content": "..."}]}`
- **Use case**: GPT-3.5/4 fine-tuning

#### Claude (Amazon Bedrock)
- **Format**: JSONL  
- **Structure**: `{"system": "...", "messages": [{"role": "user/assistant", "content": "..."}]}`
- **Use case**: Claude 3 Haiku fine-tuning on AWS Bedrock

#### Gemini (Vertex AI)
- **Format**: JSONL
- **Structure**: `{"input_text": "...", "output_text": "..."}`
- **Use case**: Gemini model fine-tuning on Google Cloud

#### Hugging Face
- **Format**: JSONL
- **Structure**: Multiple fields including `text`, `instruction`, `response`, `system`
- **Use case**: Open-source model fine-tuning

### Local/Open Source Models

#### Llama
- **Format**: JSONL
- **Structure**: `{"messages": [{"role": "system/user/assistant", "content": "..."}], "id": 0}`
- **Use case**: Llama 2/3 fine-tuning with chat template

#### Alpaca
- **Format**: JSONL
- **Structure**: `{"instruction": "...", "input": "", "output": "..."}`
- **Use case**: Stanford Alpaca format, widely supported

#### ShareGPT
- **Format**: JSONL
- **Structure**: `{"id": "...", "conversations": [{"from": "human/gpt/system", "value": "..."}]}`
- **Use case**: Conversational datasets, multi-turn dialogue

#### DeepSeek
- **Format**: JSONL
- **Structure**: `{"instruction": "...", "output": "...", "input": "", "reasoning": true}`
- **Use case**: DeepSeek-R1 style reasoning models

#### Unsloth
- **Format**: JSONL
- **Structure**: `{"text": "<|im_start|>...<|im_end|>", "conversations": [...]}`
- **Use case**: Unsloth optimized training (2x faster, 70% less memory)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Interactive Mode (Recommended)

```bash
python fine_tuning_generator.py --interactive
```

This will guide you through:
1. Selecting your target LLM provider
2. Describing your fine-tuning task
3. Setting the number of training examples
4. Choosing an output filename

### Command Line Mode

```bash
python fine_tuning_generator.py --provider openai --task "customer service chatbot" --examples 100 --output my_training_data.jsonl
```

### Arguments

- `--provider`: Target LLM provider (`openai`, `claude`, `gemini`, `huggingface`)
- `--task`: Description of your fine-tuning task
- `--examples`: Number of training examples to generate (default: 10)
- `--output`: Output filename (optional, auto-generated if not specified)
- `--interactive`: Run in interactive mode

## Examples

### Customer Service Chatbot
```bash
python fine_tuning_generator.py --provider openai --task "customer service chatbot for e-commerce" --examples 50
```

### Code Review Assistant  
```bash
python fine_tuning_generator.py --provider claude --task "code review assistant for Python" --examples 30
```

### Creative Writing Helper
```bash
python fine_tuning_generator.py --provider huggingface --task "creative writing assistant" --examples 25
```

### Local Model Examples

#### Llama Fine-tuning
```bash
python fine_tuning_generator.py --provider llama --task "math tutoring assistant" --examples 100
```

#### Alpaca Format
```bash
python fine_tuning_generator.py --provider alpaca --task "instruction following" --examples 50
```

#### DeepSeek Reasoning
```bash
python fine_tuning_generator.py --provider deepseek --task "logical reasoning problems" --examples 75
```

#### Unsloth Optimized
```bash
python fine_tuning_generator.py --provider unsloth --task "code generation" --examples 200
```

## LLM Providers for Content Generation

The tool supports multiple LLM providers for generating training content:

- **Demo Mode** (default): Uses predefined templates and patterns
- **Ollama**: Local LLM instance (requires Ollama installation)
- **Hugging Face**: Free inference API (requires API key)
- **Groq**: Fast inference API (requires API key)

## Output Examples

### OpenAI Format
```json
{"messages": [{"role": "system", "content": "You are a helpful customer service representative."}, {"role": "user", "content": "I have an issue with my order."}, {"role": "assistant", "content": "I'd be happy to help you with your order issue."}]}
```

### Claude Format
```json
{"system": "You are a helpful assistant.", "messages": [{"role": "user", "content": "How do I return an item?"}, {"role": "assistant", "content": "I can guide you through the return process."}]}
```

### Hugging Face Format
```json
{"text": "### System:\nYou are a helpful assistant.\n\n### Instruction:\nHow do I return an item?\n\n### Response:\nI can guide you through the return process.", "instruction": "How do I return an item?", "response": "I can guide you through the return process.", "system": "You are a helpful assistant."}
```

### Local Model Formats

#### Alpaca Format
```json
{"instruction": "Solve this math problem: 2x + 5 = 15", "input": "", "output": "To solve 2x + 5 = 15, subtract 5 from both sides to get 2x = 10, then divide by 2 to get x = 5."}
```

#### ShareGPT Format
```json
{"id": "sharegpt_001", "conversations": [{"from": "system", "value": "You are a helpful math tutor."}, {"from": "human", "value": "Help me solve 2x + 5 = 15"}, {"from": "gpt", "value": "I'll help you solve this step by step..."}]}
```

#### DeepSeek Format
```json
{"instruction": "Think carefully about this question and create a step-by-step chain of thoughts to ensure a logical and accurate response.\n\nSolve: 2x + 5 = 15", "output": "Let me work through this systematically...", "input": "", "reasoning": true}
```

#### Unsloth Format
```json
{"text": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nSolve 2x + 5 = 15<|im_end|>\n<|im_start|>assistant\nI'll solve this equation step by step...<|im_end|>", "conversations": [...]}
```

## Best Practices

1. **Task Description**: Be specific about your use case (e.g., "customer service for SaaS product" vs. "customer service")
2. **Example Count**: Start with 50-100 examples for initial testing, scale up based on results
3. **Quality over Quantity**: Better to have fewer high-quality examples than many generic ones
4. **Format Consistency**: Ensure your production prompts match the training format

### Local Model Recommendations

- **Alpaca**: Best for instruction-following tasks, widely compatible
- **ShareGPT**: Ideal for conversational AI and multi-turn dialogue
- **DeepSeek**: Optimized for reasoning tasks and chain-of-thought training
- **Unsloth**: Use for memory-efficient training (70% less VRAM, 2x faster)
- **Llama**: Standard chat format, good for general conversation models

### Hardware Considerations

- **Alpaca/ShareGPT**: Can run on 8-16GB VRAM
- **Unsloth**: Optimized for low VRAM (4-8GB sufficient for small models)
- **DeepSeek**: Benefits from more VRAM for reasoning tasks (16GB+)
- **QLoRA Integration**: All formats support 4-bit quantization for memory efficiency

## Requirements

- Python 3.7+
- requests (for API calls)
- Optional: Ollama (for local LLM generation)
- Optional: API keys for Hugging Face, Groq, etc.

## License

MIT License - Feel free to use and modify for your projects.