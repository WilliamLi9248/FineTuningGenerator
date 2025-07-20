#!/usr/bin/env python3
"""
Fine-Tuning Training Data Generator

A tool to generate training files for various LLM fine-tuning platforms based on user requirements.
Supports OpenAI, Claude (Bedrock), Gemini, and Hugging Face formats.
"""

import json
import csv
import argparse
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import requests
import time

@dataclass
class TrainingExample:
    """Represents a single training example"""
    system_prompt: Optional[str] = None
    user_input: str = ""
    assistant_output: str = ""
    metadata: Optional[Dict[str, Any]] = None

class LLMClient:
    """Client for various LLM APIs including free options"""
    
    def __init__(self):
        self.providers = {
            "demo": self._generate_demo,
            "ollama": self._generate_ollama,
            "openai": self._generate_openai_api,
            "claude": self._generate_claude_api,
            "huggingface": self._generate_huggingface,
            "groq": self._generate_groq
        }
    
    def generate_content(self, prompt: str, task_context: str = "", provider: str = "demo") -> str:
        """Generate content using specified LLM provider"""
        if provider not in self.providers:
            provider = "demo"
        
        return self.providers[provider](prompt, task_context)
    
    def _generate_huggingface(self, prompt: str, task_context: str) -> str:
        """Generate using Hugging Face Inference API (free tier)"""
        try:
            import os
            api_key = os.getenv('HUGGINGFACE_API_KEY')
            if not api_key:
                print("Warning: HUGGINGFACE_API_KEY not found. Using demo mode.")
                return self._generate_demo(prompt, task_context)
            
            import requests
            
            system_prompt = f"You are an expert assistant helping with: {task_context}. Provide detailed, practical, and actionable responses."
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Try different free models
            models_to_try = [
                "microsoft/DialoGPT-medium",
                "facebook/blenderbot-400M-distill", 
                "microsoft/DialoGPT-small"
            ]
            
            for model in models_to_try:
                try:
                    print(f"Generating with HuggingFace {model}")
                    response = requests.post(
                        f"https://api-inference.huggingface.co/models/{model}",
                        headers=headers,
                        json={"inputs": full_prompt, "parameters": {"max_length": 500, "temperature": 0.7}},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if isinstance(result, list) and len(result) > 0:
                            content = result[0].get('generated_text', '').replace(full_prompt, '').strip()
                            if content and len(content) > 20:
                                print(f"Success: Generated {len(content)} characters with HuggingFace")
                                return content
                    
                except Exception as e:
                    print(f"Warning: Model {model} failed: {e}")
                    continue
            
            print("Warning: All HuggingFace models failed. Using demo mode.")
            return self._generate_demo(prompt, task_context)
            
        except Exception as e:
            print(f"Error: HuggingFace generation failed: {e}")
            return self._generate_demo(prompt, task_context)
    
    def _generate_ollama(self, prompt: str, task_context: str) -> str:
        """Generate using local Ollama instance"""
        try:
            import requests
            
            # First check if Ollama is running
            try:
                health_check = requests.get("http://localhost:11434/api/tags", timeout=5)
                if health_check.status_code != 200:
                    print("Warning: Ollama server is not responding. Using demo mode.")
                    return self._generate_demo(prompt, task_context)
            except requests.exceptions.RequestException:
                print("Warning: Ollama server is not running. Please start Ollama first. Using demo mode.")
                return self._generate_demo(prompt, task_context)
            
            # Create a more effective prompt for the LLM
            system_prompt = f"You are an expert assistant helping with: {task_context}. Provide detailed, practical, and actionable responses."
            
            full_prompt = f"""System: {system_prompt}

User Query: {prompt}

Please provide a comprehensive, expert-level response that addresses the specific question with practical advice and technical details when appropriate. Be conversational but authoritative.

Response:"""

            # Try different models in order of preference
            models_to_try = ["llama3.1:8b", "llama3.1", "llama3.2", "llama2", "mistral", "codellama"]
            
            for model in models_to_try:
                try:
                    print(f"Generating with Ollama model: {model}")
                    response = requests.post("http://localhost:11434/api/generate", 
                                           json={
                                               "model": model,
                                               "prompt": full_prompt,
                                               "stream": False,
                                               "options": {
                                                   "temperature": 0.7,
                                                   "top_p": 0.9,
                                                   "max_tokens": 300
                                               }
                                           }, timeout=60)
                    
                    if response.status_code == 200:
                        result = response.json().get("response", "").strip()
                        if result and len(result) > 20:  # Ensure we got a substantial response
                            print(f"Success: Generated {len(result)} characters with {model}")
                            return result
                        else:
                            print(f"Warning: Short response from {model}, trying next model...")
                            continue
                    else:
                        print(f"Warning: Model {model} failed (status {response.status_code}), trying next...")
                        continue
                        
                except requests.exceptions.RequestException as e:
                    print(f"Warning: Model {model} failed ({str(e)}), trying next model...")
                    continue
            
            print("Warning: All Ollama models failed. Using demo mode.")
            return self._generate_demo(prompt, task_context)
            
        except Exception as e:
            print(f"Error: Ollama generation failed: {e}")
            return self._generate_demo(prompt, task_context)
    
    def _generate_openai_api(self, prompt: str, task_context: str) -> str:
        """Generate using OpenAI API"""
        try:
            import os
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                print("Warning: OPENAI_API_KEY not found. Using demo mode.")
                return self._generate_demo(prompt, task_context)
            
            import requests
            
            system_prompt = f"You are an expert assistant helping with: {task_context}. Provide detailed, practical, and actionable responses."
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }
            
            print("Generating with OpenAI GPT-3.5-turbo")
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                print(f"Success: Generated {len(content)} characters with OpenAI")
                return content
            else:
                print(f"Error: OpenAI API error: {response.status_code}")
                return self._generate_demo(prompt, task_context)
                
        except Exception as e:
            print(f"Error: OpenAI generation failed: {e}")
            return self._generate_demo(prompt, task_context)
    
    def _generate_claude_api(self, prompt: str, task_context: str) -> str:
        """Generate using Claude API"""
        try:
            import os
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                print("Warning: ANTHROPIC_API_KEY not found. Using demo mode.")
                return self._generate_demo(prompt, task_context)
            
            import requests
            
            system_prompt = f"You are an expert assistant helping with: {task_context}. Provide detailed, practical, and actionable responses."
            
            headers = {
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 500,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            
            print("Generating with Claude 3 Haiku")
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['content'][0]['text'].strip()
                print(f"Success: Generated {len(content)} characters with Claude")
                return content
            else:
                print(f"Error: Claude API error: {response.status_code}")
                return self._generate_demo(prompt, task_context)
                
        except Exception as e:
            print(f"Error: Claude generation failed: {e}")
            return self._generate_demo(prompt, task_context)
    
    def _generate_groq(self, prompt: str, task_context: str) -> str:
        """Generate using Groq API"""
        try:
            import os
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                print("Warning: GROQ_API_KEY not found. Using demo mode.")
                return self._generate_demo(prompt, task_context)
            
            import requests
            
            system_prompt = f"You are an expert assistant helping with: {task_context}. Provide detailed, practical, and actionable responses."
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "llama3-8b-8192",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }
            
            print("Generating with Groq Llama3")
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                print(f"Success: Generated {len(content)} characters with Groq")
                return content
            else:
                print(f"Error: Groq API error: {response.status_code}")
                return self._generate_demo(prompt, task_context)
                
        except Exception as e:
            print(f"Error: Groq generation failed: {e}")
            return self._generate_demo(prompt, task_context)
    
    def _generate_demo(self, prompt: str, task_context: str) -> str:
        """Demo generation with predefined responses"""
        responses = {
            "customer_service": [
                "Thank you for reaching out! I'll be glad to help resolve this issue for you right away.",
                "I understand your concern and I'm here to assist you. Let me look into this immediately.",
                "I appreciate you contacting us about this. Let me guide you through the solution step by step.",
                "I'm sorry to hear about this inconvenience. I'll make sure we get this sorted out for you quickly.",
                "Thank you for bringing this to my attention. I have the information I need to help you with this issue.",
            ],
            "code_review": [
                "This is a solid implementation! Here are some suggestions to make it even better:",
                "I've analyzed your code and identified several optimization opportunities:",
                "The core logic works well. Let me suggest some improvements for performance and readability:",
                "Great approach! Here are a few best practices that could enhance this code further:",
                "This code demonstrates good understanding. Consider these refinements for production use:",
            ],
            "creative_writing": [
                "Once upon a time, in a realm where dreams took physical form, there lived a weaver of stories who...",
                "The crimson sunset painted the sky as Elena discovered the hidden journal that would unveil her family's secret...",
                "In the bustling marketplace of New Terra, where technology and magic coexisted, Marcus stumbled upon an ancient artifact...",
                "The lighthouse keeper had seen many storms, but none quite like the one approaching from the temporal rift...",
                "As the last dragon of the northern mountains prepared for eternal sleep, she entrusted her most precious secret to...",
            ],
            "technical_explanation": [
                "Let me break this down into clear, manageable steps that build on each other:",
                "This concept is fundamental to understanding the broader system. Here's how it works in practice:",
                "I'll explain this using a practical example that demonstrates the key principles:",
                "To grasp this fully, let's start with the basics and work our way up to the complex parts:",
                "Think of this as a puzzle where each piece represents a different aspect of the solution:",
            ]
        }
        
        # Enhanced context-aware selection
        context_lower = (task_context + " " + prompt).lower()
        for key, response_list in responses.items():
            if key in context_lower:
                import random
                return random.choice(response_list)
        
        # If no specific category matches, use enhanced demo
        return self._generate_enhanced_demo(prompt, task_context)
    
    def _generate_enhanced_demo(self, prompt: str, task_context: str) -> str:
        """Enhanced demo generation with content-specific responses"""
        import random
        
        # Extract key elements from the prompt to make responses more specific
        prompt_lower = prompt.lower()
        context_lower = task_context.lower()
        
        # Generate content-aware responses based on specific prompt content
        if "order" in prompt_lower and ("issue" in prompt_lower or "problem" in prompt_lower):
            return random.choice([
                "I understand you're having trouble with your order. Let me pull up your account details and see what's happening. Can you provide your order number?",
                "I'm sorry to hear about the issue with your order. I'll investigate this right away and work to resolve it quickly for you.",
                "Thank you for reaching out about your order concern. Let me check the status and see exactly what's going on."
            ])
        elif "return" in prompt_lower or "exchange" in prompt_lower:
            return random.choice([
                "I'd be happy to help you with your return. Our return process is straightforward - items can be returned within 30 days with the original receipt.",
                "Returning an item is easy! Let me guide you through our return policy and the steps you'll need to follow.",
                "I can definitely assist with processing your return. First, let me check if your item is eligible and walk you through the next steps."
            ])
        elif "account" in prompt_lower and "locked" in prompt_lower:
            return random.choice([
                "I see your account is locked. This usually happens for security reasons. Let me help you regain access - I'll need to verify your identity first.",
                "Account lockouts can be frustrating! Let me assist you with unlocking your account. I'll guide you through our security verification process.",
                "I understand how inconvenient a locked account can be. I'm here to help restore your access safely and securely."
            ])
        elif "charged" in prompt_lower and ("twice" in prompt_lower or "double" in prompt_lower):
            return random.choice([
                "I apologize for the billing error. Double charges can occur due to processing delays. Let me investigate this charge and arrange a refund if needed.",
                "I see you were charged twice - that's definitely not right. Let me look into this billing issue and get it corrected for you immediately.",
                "Double billing is unacceptable, and I'll fix this right away. I'm checking your payment history now to resolve this error."
            ])
        elif "code" in prompt_lower or "function" in prompt_lower or "algorithm" in prompt_lower:
            return random.choice([
                "I'll review your code carefully and provide specific suggestions for improvement, focusing on performance, readability, and best practices.",
                "Looking at your implementation, I can identify several optimization opportunities and recommend proven patterns for better code quality.",
                "I'll analyze your code structure and logic, then suggest concrete improvements that will make it more efficient and maintainable."
            ])
        elif "debug" in prompt_lower or "error" in prompt_lower:
            return random.choice([
                "I'll help you identify the root cause of this error. Let's start by examining the error message and tracing through the problematic code section.",
                "Debugging can be tricky, but I'll guide you through a systematic approach to isolate and fix the issue you're encountering.",
                "I'll assist you in diagnosing this problem step-by-step, using proven debugging techniques to identify and resolve the error."
            ])
        elif "sql" in prompt_lower or "query" in prompt_lower or "database" in prompt_lower:
            return random.choice([
                "I'll analyze your SQL query and identify performance bottlenecks, then suggest specific optimizations like proper indexing and query restructuring.",
                "Let me examine your query structure and recommend improvements for better execution time, including index optimization and query rewriting.",
                "I'll help you optimize this database query by analyzing the execution plan and suggesting more efficient approaches."
            ])
        elif "security" in prompt_lower or "vulnerability" in prompt_lower:
            return random.choice([
                "I'll conduct a thorough security review of your code, checking for common vulnerabilities like injection attacks, authentication flaws, and data exposure risks.",
                "Let me examine your code for potential security issues, focusing on input validation, authentication, authorization, and data protection.",
                "I'll help identify security vulnerabilities in your implementation and recommend secure coding practices to mitigate these risks."
            ])
        elif "test" in prompt_lower or "testing" in prompt_lower:
            return random.choice([
                "I'll design a comprehensive testing strategy covering unit tests, integration tests, and edge cases to ensure your function works reliably.",
                "Let me outline a thorough testing approach that includes test case design, mock strategies, and validation methods for this complex function.",
                "I'll help you create an effective test suite with proper coverage, including boundary conditions and error scenarios."
            ])
        elif "package" in prompt_lower or "delivery" in prompt_lower or "shipping" in prompt_lower:
            return random.choice([
                "I understand your concern about the delayed package. Let me track your shipment immediately and provide you with an updated delivery status.",
                "I'll investigate your shipment tracking right away and find out exactly where your package is and when you can expect it.",
                "Let me check the shipping status for you and see what's causing the delay. I'll also explore options to expedite delivery if possible."
            ])
        elif "warranty" in prompt_lower or "repair" in prompt_lower:
            return random.choice([
                "I'm sorry your product failed so quickly. Let me check your warranty coverage and arrange for a replacement or repair at no cost to you.",
                "That's definitely not acceptable for such a new product. I'll review your warranty terms and get this resolved with either a repair or replacement.",
                "I understand your frustration with this early failure. Let me verify your warranty status and initiate the appropriate resolution process."
            ])
        elif "compatible" in prompt_lower or "compatibility" in prompt_lower:
            return random.choice([
                "I'll help you determine compatibility by reviewing the technical specifications and requirements for both your current setup and this product.",
                "Let me check the compatibility requirements for you. I'll need some details about your current system to ensure everything will work together properly.",
                "I can definitely help verify compatibility. Let me walk you through the key specifications and requirements to make sure this is the right fit."
            ])
        elif "story" in prompt_lower or "character" in prompt_lower or "narrative" in prompt_lower:
            return random.choice([
                "I'll craft an engaging narrative with vivid characters and compelling plot elements that bring your story idea to life.",
                "Let me create a rich, detailed story that captures the essence of your vision with authentic dialogue and immersive settings.",
                "I'll develop this narrative concept with well-rounded characters and engaging storytelling techniques that draw readers in."
            ])
        elif "poem" in prompt_lower or "poetry" in context_lower:
            return random.choice([
                "I'll compose a lyrical poem that captures the emotions and imagery you're looking for, using rhythm and metaphor effectively.",
                "Let me create verses that resonate with your theme, employing poetic devices to create beauty and meaning.",
                "I'll craft a poem that balances form and content, using language that evokes the feelings and scenes you envision."
            ])
        elif "math" in context_lower or "solve" in prompt_lower or "equation" in prompt_lower:
            return random.choice([
                "I'll solve this step-by-step, clearly explaining each mathematical operation so you can understand the process and apply it to similar problems.",
                "Let me break down this problem systematically, showing you the underlying mathematical principles and solution methodology.",
                "I'll work through this calculation methodically, highlighting key concepts and strategies you can use for future problems."
            ])
        elif "franka" in prompt_lower or ("robot" in context_lower and "cooking" in context_lower):
            return random.choice([
                "For the Franka Panda in cooking tasks, I recommend implementing a shared control architecture where the robot monitors human actions and adapts its movements accordingly. Key considerations include force-torque sensing for safe physical interaction, visual tracking of human hand positions, and predefined cooking primitives like stirring, pouring, and chopping that can be triggered contextually.",
                "The Franka Panda's 7 DOF gives us excellent dexterity for kitchen tasks. I suggest designing the interaction protocol with three layers: 1) High-level task coordination (who does what when), 2) Real-time motion adaptation (collision avoidance, shared workspace management), and 3) Low-level safety monitoring (force limits, emergency stops). The robot should use its compliant control to safely work alongside humans.",
                "For breakfast cooking collaboration, consider implementing a state machine that tracks the cooking process (prep, cooking, plating) and assigns appropriate roles to human and robot. The Franka should use impedance control for safe physical interaction, computer vision for ingredient recognition, and natural language processing for voice commands. Start with simple handoff scenarios like 'robot holds bowl while human mixes'."
            ])
        elif "safety" in prompt_lower and ("robot" in context_lower or "franka" in context_lower):
            return random.choice([
                "Safety in human-robot cooking collaboration requires multiple layers: 1) Physical safety through compliant control and force limiting, 2) Workspace safety with collision detection and avoidance, 3) Food safety with proper sanitation protocols, and 4) Operational safety with clear communication of robot intentions. The Franka's built-in safety features should be configured for kitchen environments with appropriate force thresholds.",
                "Key safety considerations include: Setting appropriate joint torque limits for safe contact, implementing workspace monitoring to detect human presence, using the Franka's external force estimation for collision detection, maintaining safe velocities near humans, and ensuring emergency stop accessibility. I also recommend visual and audio cues to communicate robot intentions to the human collaborator.",
                "For kitchen safety with the Franka Panda: Configure the robot's safety-rated monitored stop for immediate halting when humans enter critical zones, implement soft collision detection using the arm's torque sensors, use predictive motion planning to avoid potential collisions, and establish clear protocols for tool handoffs and shared workspace access. The robot should always yield right-of-way to human movements."
            ])
        elif "motion planning" in prompt_lower or ("programming" in prompt_lower and "robot" in context_lower):
            return random.choice([
                "For adaptive motion planning with the Franka Panda during cooking, implement a hierarchical approach: Use high-level task planning to sequence cooking actions, medium-level path planning with dynamic obstacle avoidance for human movements, and low-level impedance control for compliant interaction. The robot should continuously update its plans based on human actions and cooking state changes.",
                "I recommend implementing reactive motion planning that combines predictive human motion models with real-time replanning. Use the Franka's joint position and velocity feedback along with external sensors (cameras, force sensors) to track human movements and adapt robot trajectories. Implement shared control where the human can guide the robot through physical interaction while the robot maintains safety constraints.",
                "For breakfast preparation, the Franka should use context-aware motion planning that considers both the current cooking state and human intentions. Implement behavior trees that can switch between different motion primitives (reaching, stirring, pouring) based on cooking context. Use machine learning to adapt to individual human preferences and working styles over time."
            ])
        elif "communication" in prompt_lower and "robot" in context_lower:
            return random.choice([
                "Effective human-robot communication during cooking should be multimodal: Use visual cues (LED indicators, screen displays) to show robot status and intentions, audio feedback for confirmations and alerts, haptic communication through compliant physical interaction, and gesture recognition for natural command input. The robot should provide clear feedback about what it's doing and what it needs from the human.",
                "I suggest implementing a natural communication protocol that includes: Voice commands for high-level task coordination ('start mixing', 'hand me the spatula'), gesture recognition for intuitive interaction (pointing, hand signals), visual display of robot plans and status, and force-based communication during physical collaboration. The system should learn individual communication preferences over time.",
                "For intuitive cooking collaboration, design the communication system around cooking workflow: The robot should announce its intentions before acting ('I'll hold the bowl now'), request assistance when needed ('Please add the flour'), provide status updates during long tasks ('Still mixing, 30 seconds remaining'), and use physical compliance to communicate readiness for handoffs. Keep communication simple and cooking-context relevant."
            ])
        elif "robot" in context_lower or "robotic" in context_lower or "automation" in context_lower:
            return random.choice([
                "I'll help you design effective human-robot interaction patterns, focusing on intuitive communication protocols and collaborative workflows.",
                "Let me suggest approaches for seamless human-robot collaboration, considering safety protocols and efficiency optimization.",
                "I'll guide you through implementing robust interaction systems that enable natural and productive human-robot teamwork."
            ])
        elif "brainstorm" in prompt_lower or "ideas" in prompt_lower:
            return random.choice([
                "I'd love to brainstorm with you! Let me share some innovative approaches and creative solutions to spark new ideas.",
                "Let's explore this creatively together. I'll offer fresh perspectives and unique angles to help generate breakthrough concepts.",
                "I'll help you think outside the box with diverse ideas and innovative approaches tailored to your specific goals."
            ])
        else:
            # Fallback with more specific language based on context
            if "guidance" in prompt_lower:
                return random.choice([
                    "I'll provide comprehensive guidance tailored to your specific situation, breaking down complex concepts into actionable steps.",
                    "Let me offer detailed direction based on proven methodologies and best practices in this area.",
                    "I'll guide you through this systematically, ensuring you have clear understanding and practical next steps."
                ])
            elif "help" in prompt_lower or "assist" in prompt_lower:
                return random.choice([
                    "I'm here to provide hands-on assistance with your specific needs, offering practical solutions and expert insights.",
                    "I'll help you tackle this challenge effectively, drawing on relevant expertise and proven strategies.",
                    "Let me assist you with a tailored approach that addresses your unique requirements and objectives."
                ])
            else:
                return random.choice([
                    "I'll address this comprehensively, providing detailed insights and practical recommendations based on your specific context.",
                    "Let me offer a thorough analysis with actionable solutions that directly address your particular situation.",
                    "I'll provide expert guidance tailored specifically to your needs, ensuring you get the most relevant and useful assistance."
                ])

class FineTuningGenerator:
    """Main class for generating fine-tuning training data"""
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.supported_providers = {
            "openai": self._format_openai,
            "claude": self._format_claude,
            "gemini": self._format_gemini,
            "huggingface": self._format_huggingface,
            "llama": self._format_llama,
            "alpaca": self._format_alpaca,
            "sharegpt": self._format_sharegpt,
            "deepseek": self._format_deepseek,
            "unsloth": self._format_unsloth
        }
    
    def generate_training_data(self, 
                             provider: str, 
                             task_description: str, 
                             num_examples: int = 10,
                             output_file: str = None,
                             llm_provider: str = "ollama") -> str:
        """Generate training data for the specified provider"""
        
        if provider not in self.supported_providers:
            raise ValueError(f"Unsupported provider: {provider}. Supported: {list(self.supported_providers.keys())}")
        
        print(f"Generating {num_examples} training examples for {provider}...")
        
        # Generate examples based on task description
        examples = self._generate_examples(task_description, num_examples, llm_provider)
        
        # Format according to provider requirements
        formatted_data = self.supported_providers[provider](examples)
        
        # Determine output file if not specified
        if not output_file:
            extension = self._get_file_extension(provider)
            output_file = f"training_data_{provider}_{int(time.time())}.{extension}"
        
        # Write to file
        self._write_to_file(formatted_data, output_file, provider)
        
        return output_file
    
    def _generate_examples(self, task_description: str, num_examples: int, llm_provider: str = "demo") -> List[TrainingExample]:
        """Generate training examples based on task description"""
        examples = []
        
        # Create diverse examples based on the task
        example_templates = self._create_example_templates(task_description)
        
        for i in range(num_examples):
            template = example_templates[i % len(example_templates)]
            
            # Generate content using LLM
            user_input = template["user_template"].format(index=i+1)
            assistant_output = self.llm_client.generate_content(user_input, task_description, llm_provider)
            
            example = TrainingExample(
                system_prompt=template.get("system_prompt"),
                user_input=user_input,
                assistant_output=assistant_output,
                metadata={"example_id": i+1, "task": task_description}
            )
            examples.append(example)
        
        return examples
    
    def _analyze_task_context(self, task_description: str) -> Dict[str, Any]:
        """Analyze the task description to understand the specific domain and requirements"""
        task_lower = task_description.lower()
        
        # Extract key information from the task description
        context = {
            "domain": "general",
            "key_entities": [],
            "specific_tasks": [],
            "expertise_level": "intermediate"
        }
        
        # Robotics context
        if any(word in task_lower for word in ["robot", "robotic", "franka", "panda", "arm", "dof"]):
            context["domain"] = "robotics"
            if "franka" in task_lower or "panda" in task_lower:
                context["key_entities"].append("Franka Emika Panda")
            if "cooking" in task_lower or "breakfast" in task_lower:
                context["specific_tasks"].append("cooking collaboration")
            if "interaction" in task_lower or "collaborate" in task_lower:
                context["specific_tasks"].append("human-robot interaction")
        
        # Customer service context
        elif any(word in task_lower for word in ["customer", "support", "service"]):
            context["domain"] = "customer_service"
            if "e-commerce" in task_lower or "shopping" in task_lower:
                context["specific_tasks"].append("e-commerce support")
        
        # Programming context
        elif any(word in task_lower for word in ["code", "programming", "software", "debug"]):
            context["domain"] = "programming"
            if "python" in task_lower:
                context["key_entities"].append("Python")
            if "web" in task_lower:
                context["specific_tasks"].append("web development")
        
        # Math/Education context
        elif any(word in task_lower for word in ["math", "tutoring", "teaching", "education"]):
            context["domain"] = "education"
            if "math" in task_lower:
                context["specific_tasks"].append("mathematics")
        
        return context

    def _create_robotics_templates(self, task_description: str, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Create robotics-specific conversation templates"""
        if "Franka Emika Panda" in context["key_entities"] and "cooking collaboration" in context["specific_tasks"]:
            return [
                {
                    "system_prompt": "You are a robotics engineer specializing in human-robot collaboration systems, particularly with Franka Emika Panda arms.",
                    "user_template": "I'm setting up the Franka Panda for breakfast cooking tasks. How should I design the interaction protocol between the human chef and robot?"
                },
                {
                    "system_prompt": "You are an expert in collaborative robotics and safe human-robot interaction during food preparation tasks.",
                    "user_template": "What safety considerations should I implement when the Franka arm is working alongside humans in a kitchen environment?"
                },
                {
                    "system_prompt": "You are a motion planning specialist who works with 7-DOF robotic arms in dynamic environments.",
                    "user_template": "How can I program the Franka Panda to adaptively respond to human movements while preparing breakfast?"
                },
                {
                    "system_prompt": "You are a human-robot interaction researcher focused on intuitive communication protocols.",
                    "user_template": "What's the best way to implement natural communication between the human and robot during cooking tasks?"
                },
                {
                    "system_prompt": "You are a robotics integration engineer with expertise in kitchen automation and collaborative cooking.",
                    "user_template": "I need help fine-tuning the robot's responses to human actions during breakfast preparation. What approach would you recommend?"
                }
            ]
        else:
            # Generic robotics templates
            return [
                {
                    "system_prompt": "You are a robotics engineer with expertise in robotic arm control and automation.",
                    "user_template": "I'm working on a robotic arm project. Can you help me with the control algorithms?"
                },
                {
                    "system_prompt": "You are a specialist in human-robot interaction and collaborative robotics systems.",
                    "user_template": "How can I improve the interaction between humans and robots in my application?"
                }
            ]

    def _create_example_templates(self, task_description: str) -> List[Dict[str, str]]:
        """Create realistic conversation templates based on analyzed task context"""
        context = self._analyze_task_context(task_description)
        task_lower = task_description.lower()
        
        if context["domain"] == "robotics":
            return self._create_robotics_templates(task_description, context)
        elif context["domain"] == "customer_service":
            return [
                {
                    "system_prompt": "You are a friendly and knowledgeable customer service representative dedicated to resolving customer issues efficiently.",
                    "user_template": "I have an issue with my order #{index}. Can you help?"
                },
                {
                    "system_prompt": "You are an experienced support agent specializing in returns and exchanges.",
                    "user_template": "How do I return an item I purchased?"
                },
                {
                    "system_prompt": "You are a patient technical support specialist who helps customers with account access issues.",
                    "user_template": "My account is locked and I can't log in."
                },
                {
                    "system_prompt": "You are a helpful customer care representative focused on billing and payment inquiries.",
                    "user_template": "I was charged twice for the same order. What happened?"
                },
                {
                    "system_prompt": "You are a proactive customer success specialist who ensures customer satisfaction.",
                    "user_template": "I'm not happy with my recent purchase. What are my options?"
                },
                {
                    "system_prompt": "You are a knowledgeable shipping specialist who handles delivery inquiries.",
                    "user_template": "My package was supposed to arrive yesterday but I still haven't received it. Where is it?"
                },
                {
                    "system_prompt": "You are a product information expert who helps customers understand features and compatibility.",
                    "user_template": "Can you help me understand if this product is compatible with my current setup?"
                },
                {
                    "system_prompt": "You are a warranty and repair specialist who assists with product issues.",
                    "user_template": "My product stopped working after just 2 weeks. Is this covered under warranty?"
                }
            ]
        
        elif "code" in task_lower or "programming" in task_lower:
            return [
                {
                    "system_prompt": "You are a senior software engineer with expertise in code optimization and best practices.",
                    "user_template": "Can you review this Python function and suggest improvements?"
                },
                {
                    "system_prompt": "You are a performance optimization specialist who helps developers write efficient code.",
                    "user_template": "How do I optimize this algorithm for better performance?"
                },
                {
                    "system_prompt": "You are an experienced developer focused on robust error handling and defensive programming.",
                    "user_template": "What's the best way to handle errors in this code?"
                },
                {
                    "system_prompt": "You are a code architecture expert who designs scalable and maintainable systems.",
                    "user_template": "How should I structure this application for better maintainability?"
                },
                {
                    "system_prompt": "You are a debugging specialist who excels at identifying and fixing complex issues.",
                    "user_template": "I'm getting an unexpected error in my code. Can you help me debug it?"
                },
                {
                    "system_prompt": "You are a database optimization expert who specializes in efficient query design.",
                    "user_template": "My SQL query is running too slowly. How can I make it faster?"
                },
                {
                    "system_prompt": "You are a security-focused developer who identifies and prevents vulnerabilities.",
                    "user_template": "Can you help me identify potential security issues in this code?"
                },
                {
                    "system_prompt": "You are a testing specialist who creates comprehensive test strategies.",
                    "user_template": "What's the best approach for testing this complex function?"
                }
            ]
        
        elif "creative" in task_lower or "writing" in task_lower:
            return [
                {
                    "system_prompt": "You are an imaginative storyteller who crafts engaging narratives with vivid details and compelling characters.",
                    "user_template": "Write a short story about a time traveler who visits the year {index}030."
                },
                {
                    "system_prompt": "You are a lyrical poet who specializes in nature imagery and emotional expression.",
                    "user_template": "Create a poem about the changing seasons."
                },
                {
                    "system_prompt": "You are a dialogue specialist who creates authentic conversations that reveal character personalities.",
                    "user_template": "Write a dialogue between two characters meeting for the first time."
                },
                {
                    "system_prompt": "You are a creative writing mentor who helps develop compelling fictional scenarios and plots.",
                    "user_template": "Help me brainstorm ideas for a mystery novel set in a small town."
                },
                {
                    "system_prompt": "You are an expert in character development who creates multi-dimensional, relatable protagonists.",
                    "user_template": "Describe a complex character who is both a hero and an anti-hero."
                }
            ]
        
        else:
            return [
                {
                    "system_prompt": f"You are a knowledgeable specialist in {task_description} who provides detailed, actionable guidance.",
                    "user_template": f"Please help me with this {task_description} task #{'{index}'}."
                },
                {
                    "system_prompt": f"You are an experienced professional with deep expertise in {task_description} and a track record of solving complex problems.",
                    "user_template": f"Can you provide guidance on {task_description}?"
                },
                {
                    "system_prompt": f"You are a patient and thorough mentor who excels at teaching {task_description} concepts clearly.",
                    "user_template": f"I need assistance with {task_description}. What should I do?"
                },
                {
                    "system_prompt": f"You are a results-oriented consultant specializing in {task_description} with a focus on practical solutions.",
                    "user_template": f"What's the best approach for handling {task_description} challenges?"
                },
                {
                    "system_prompt": f"You are an innovative problem-solver who brings creative approaches to {task_description} tasks.",
                    "user_template": f"I'm looking for fresh ideas about {task_description}. Can you help brainstorm?"
                }
            ]
    
    def _format_openai(self, examples: List[TrainingExample]) -> List[Dict[str, Any]]:
        """Format data for OpenAI fine-tuning (JSONL)"""
        formatted = []
        for example in examples:
            messages = []
            
            if example.system_prompt:
                messages.append({"role": "system", "content": example.system_prompt})
            
            messages.append({"role": "user", "content": example.user_input})
            messages.append({"role": "assistant", "content": example.assistant_output})
            
            formatted.append({"messages": messages})
        
        return formatted
    
    def _format_claude(self, examples: List[TrainingExample]) -> List[Dict[str, Any]]:
        """Format data for Claude fine-tuning on Bedrock (JSONL)"""
        formatted = []
        for example in examples:
            messages = [
                {"role": "user", "content": example.user_input},
                {"role": "assistant", "content": example.assistant_output}
            ]
            
            data = {"messages": messages}
            if example.system_prompt:
                data["system"] = example.system_prompt
            
            formatted.append(data)
        
        return formatted
    
    def _format_gemini(self, examples: List[TrainingExample]) -> List[Dict[str, Any]]:
        """Format data for Gemini fine-tuning (JSONL)"""
        formatted = []
        for example in examples:
            # Gemini format can vary, using a common conversational format
            input_text = f"Instruction: {example.user_input}"
            if example.system_prompt:
                input_text = f"System: {example.system_prompt}\n{input_text}"
            
            formatted.append({
                "input_text": input_text,
                "output_text": example.assistant_output
            })
        
        return formatted
    
    def _format_huggingface(self, examples: List[TrainingExample]) -> List[Dict[str, Any]]:
        """Format data for Hugging Face fine-tuning (JSONL)"""
        formatted = []
        for example in examples:
            # Hugging Face flexible format
            text = f"### Instruction:\n{example.user_input}\n\n### Response:\n{example.assistant_output}"
            if example.system_prompt:
                text = f"### System:\n{example.system_prompt}\n\n{text}"
            
            formatted.append({
                "text": text,
                "instruction": example.user_input,
                "response": example.assistant_output,
                "system": example.system_prompt or "",
            })
        
        return formatted
    
    def _format_llama(self, examples: List[TrainingExample]) -> List[Dict[str, Any]]:
        """Format data for Llama fine-tuning (Chat template format)"""
        formatted = []
        for example in examples:
            messages = []
            
            if example.system_prompt:
                messages.append({"role": "system", "content": example.system_prompt})
            
            messages.append({"role": "user", "content": example.user_input})
            messages.append({"role": "assistant", "content": example.assistant_output})
            
            # Llama chat template format
            formatted.append({
                "messages": messages,
                "id": example.metadata.get("example_id", 0) if example.metadata else 0
            })
        
        return formatted
    
    def _format_alpaca(self, examples: List[TrainingExample]) -> List[Dict[str, Any]]:
        """Format data for Alpaca fine-tuning format"""
        formatted = []
        for example in examples:
            data = {
                "instruction": example.user_input,
                "input": "",  # Alpaca uses empty input for simple instruction-following
                "output": example.assistant_output
            }
            
            # If there's a system prompt, include it in the instruction
            if example.system_prompt:
                data["instruction"] = f"{example.system_prompt}\n\n{example.user_input}"
            
            formatted.append(data)
        
        return formatted
    
    def _format_sharegpt(self, examples: List[TrainingExample]) -> List[Dict[str, Any]]:
        """Format data for ShareGPT format (conversational)"""
        formatted = []
        for example in examples:
            conversations = []
            
            if example.system_prompt:
                conversations.append({"from": "system", "value": example.system_prompt})
            
            conversations.append({"from": "human", "value": example.user_input})
            conversations.append({"from": "gpt", "value": example.assistant_output})
            
            formatted.append({
                "id": f"sharegpt_{example.metadata.get('example_id', 0)}" if example.metadata else "sharegpt_0",
                "conversations": conversations
            })
        
        return formatted
    
    def _format_deepseek(self, examples: List[TrainingExample]) -> List[Dict[str, Any]]:
        """Format data for DeepSeek fine-tuning"""
        formatted = []
        for example in examples:
            # DeepSeek prefers instruction-response format with reasoning
            instruction = example.user_input
            if example.system_prompt:
                instruction = f"Context: {example.system_prompt}\n\nInstruction: {example.user_input}"
            
            # Add reasoning prompt for DeepSeek-R1 style
            reasoning_prompt = "Think carefully about this question and create a step-by-step chain of thoughts to ensure a logical and accurate response."
            
            formatted.append({
                "instruction": f"{reasoning_prompt}\n\n{instruction}",
                "output": example.assistant_output,
                "input": "",
                "reasoning": True
            })
        
        return formatted
    
    def _format_unsloth(self, examples: List[TrainingExample]) -> List[Dict[str, Any]]:
        """Format data for Unsloth fine-tuning (optimized chat format)"""
        formatted = []
        for example in examples:
            # Unsloth uses optimized chat template
            conversation = ""
            
            if example.system_prompt:
                conversation += f"<|im_start|>system\n{example.system_prompt}<|im_end|>\n"
            
            conversation += f"<|im_start|>user\n{example.user_input}<|im_end|>\n"
            conversation += f"<|im_start|>assistant\n{example.assistant_output}<|im_end|>"
            
            formatted.append({
                "text": conversation,
                "conversations": [
                    {"role": "system", "content": example.system_prompt or ""},
                    {"role": "user", "content": example.user_input},
                    {"role": "assistant", "content": example.assistant_output}
                ]
            })
        
        return formatted
    
    def _get_file_extension(self, provider: str) -> str:
        """Get appropriate file extension for provider"""
        return "jsonl"  # All providers currently use JSONL
    
    def _write_to_file(self, data: List[Dict[str, Any]], filename: str, provider: str):
        """Write formatted data to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Training data written to {filename}")
        print(f"Format: {provider.upper()}")
        print(f"Examples: {len(data)}")

def interactive_mode():
    """Interactive mode for generating training data"""
    print("Fine-Tuning Training Data Generator")
    print("=" * 50)
    
    generator = FineTuningGenerator()
    
    # Get provider
    print("\nSupported providers:")
    print("Commercial/Cloud:")
    commercial = ["openai", "claude", "gemini", "huggingface"]
    for i, provider in enumerate(commercial, 1):
        print(f"  {i}. {provider.upper()}")
    
    print("\nLocal/Open Source:")
    local = ["llama", "alpaca", "sharegpt", "deepseek", "unsloth"]
    for i, provider in enumerate(local, len(commercial) + 1):
        print(f"  {i}. {provider.upper()}")
    
    all_providers = commercial + local
    
    while True:
        try:
            choice = input(f"\nSelect provider (1-{len(all_providers)}): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(all_providers):
                provider = all_providers[int(choice) - 1]
                break
            else:
                print("Invalid choice. Please try again.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)
    
    # Get task description
    print(f"\nSelected: {provider.upper()}")
    task_description = input("Describe your fine-tuning task (e.g., 'customer service chatbot', 'code review assistant'): ").strip()
    
    if not task_description:
        print("Task description is required!")
        return
    
    # Get number of examples
    while True:
        try:
            num_examples = input("Number of training examples to generate (default: 10): ").strip()
            if not num_examples:
                num_examples = 10
            else:
                num_examples = int(num_examples)
            
            if num_examples < 1:
                print("Number of examples must be at least 1.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")
    
    # Get LLM provider
    print("\nLLM providers for content generation:")
    llm_options = [
        ("demo", "Pre-built expert responses (fast, consistent)"),
        ("ollama", "Local Ollama (free, requires setup)"),
        ("openai", "OpenAI API (requires API key)"),
        ("claude", "Claude API (requires API key)"),
        ("huggingface", "Hugging Face (free tier)"),
        ("groq", "Groq API (fast, requires API key)")
    ]
    
    for i, (llm_provider_name, desc) in enumerate(llm_options, 1):
        print(f"  {i}. {llm_provider_name.upper()}: {desc}")
    
    while True:
        try:
            llm_choice = input(f"\nSelect LLM provider (1-{len(llm_options)}, default: 1 for demo): ").strip()
            if not llm_choice:
                llm_provider = "demo"
                break
            elif llm_choice.isdigit() and 1 <= int(llm_choice) <= len(llm_options):
                llm_provider = llm_options[int(llm_choice) - 1][0]
                break
            else:
                print("Invalid choice. Please try again.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)
    
    # Get output filename (optional)
    output_file = input("Output filename (optional, will auto-generate if empty): ").strip()
    if not output_file:
        output_file = None
    
    # Generate training data
    print("\n" + "=" * 50)
    try:
        output_path = generator.generate_training_data(
            provider=provider,
            task_description=task_description,
            num_examples=num_examples,
            output_file=output_file,
            llm_provider=llm_provider
        )
        
        print(f"\nSuccess: Successfully generated training data!")
        print(f"File: {output_path}")
        print(f"Format: {provider.upper()}")
        print(f"LLM: {llm_provider.upper()}")
        print(f"Examples: {num_examples}")
        
        # Show preview
        if input("\nWould you like to see a preview? (y/n): ").lower().startswith('y'):
            print("\n" + "=" * 30 + " PREVIEW " + "=" * 30)
            with open(output_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 2:  # Show first 2 examples
                        break
                    example = json.loads(line)
                    print(f"Example {i+1}:")
                    print(json.dumps(example, indent=2, ensure_ascii=False))
                    print("-" * 60)
        
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate fine-tuning training data for various LLM providers")
    parser.add_argument("--provider", choices=["openai", "claude", "gemini", "huggingface", "llama", "alpaca", "sharegpt", "deepseek", "unsloth"], 
                       help="Target LLM provider")
    parser.add_argument("--task", help="Description of the fine-tuning task")
    parser.add_argument("--examples", type=int, default=10, help="Number of examples to generate")
    parser.add_argument("--output", help="Output filename")
    parser.add_argument("--llm", choices=["demo", "ollama", "openai", "claude", "huggingface", "groq"], default="demo",
                       help="LLM provider for content generation: demo (pre-built), ollama (free local), openai/claude (API), huggingface (free), groq (API)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    if args.interactive or not all([args.provider, args.task]):
        interactive_mode()
    else:
        generator = FineTuningGenerator()
        output_file = generator.generate_training_data(
            provider=args.provider,
            task_description=args.task,
            num_examples=args.examples,
            output_file=args.output,
            llm_provider=args.llm
        )
        print(f"Training data generated: {output_file}")

if __name__ == "__main__":
    main()