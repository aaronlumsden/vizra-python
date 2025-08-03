"""
Production-ready example of an AI Art Generation Agent with real API integration.
"""

import os
import json
import requests
from typing import Optional, Dict, Any
from vizra import BaseAgent, ToolInterface, AgentContext

# You'll need to install: pip install openai pillow requests


class ProductionImageGenerationTool(ToolInterface):
    """
    Production-ready tool for generating images using OpenAI's DALL-E API.
    """
    
    def __init__(self):
        # Initialize OpenAI client
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            # For OpenPipe, you might use:
            # base_url="https://api.openpipe.ai/v1",
            # api_key=os.getenv('OPENPIPE_API_KEY'),
        )
    
    def definition(self) -> dict:
        return {
            'name': 'generate_image',
            'description': 'Generate an image using DALL-E based on a text description',
            'parameters': {
                'type': 'object',
                'properties': {
                    'prompt': {
                        'type': 'string',
                        'description': 'Detailed description of the image to generate (max 4000 chars)'
                    },
                    'quality': {
                        'type': 'string',
                        'description': 'Image quality',
                        'enum': ['standard', 'hd'],
                        'default': 'standard'
                    },
                    'size': {
                        'type': 'string',
                        'description': 'Image size',
                        'enum': ['1024x1024', '1024x1792', '1792x1024'],
                        'default': '1024x1024'
                    },
                    'style': {
                        'type': 'string',
                        'description': 'Image style',
                        'enum': ['vivid', 'natural'],
                        'default': 'vivid'
                    }
                },
                'required': ['prompt']
            }
        }
    
    def execute(self, arguments: dict, context: AgentContext) -> str:
        """Generate an image using DALL-E API."""
        try:
            # Make API call to DALL-E
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=arguments['prompt'],
                size=arguments.get('size', '1024x1024'),
                quality=arguments.get('quality', 'standard'),
                style=arguments.get('style', 'vivid'),
                n=1
            )
            
            image_url = response.data[0].url
            revised_prompt = response.data[0].revised_prompt
            
            # Optionally download and save the image
            image_data = requests.get(image_url).content
            filename = f"generated_{hash(arguments['prompt'])}.png"
            
            # Save locally (optional)
            # with open(filename, 'wb') as f:
            #     f.write(image_data)
            
            return json.dumps({
                'status': 'success',
                'image_url': image_url,
                'revised_prompt': revised_prompt,
                'original_prompt': arguments['prompt'],
                'size': arguments.get('size', '1024x1024'),
                'filename': filename
            })
            
        except Exception as e:
            return json.dumps({
                'status': 'error',
                'error': str(e),
                'message': 'Failed to generate image. Please check your API key and try again.'
            })


class VisionAnalysisTool(ToolInterface):
    """
    Tool for analyzing images using GPT-4 Vision.
    """
    
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def definition(self) -> dict:
        return {
            'name': 'analyze_image',
            'description': 'Analyze an image using GPT-4 Vision',
            'parameters': {
                'type': 'object',
                'properties': {
                    'image_url': {
                        'type': 'string',
                        'description': 'URL of the image to analyze'
                    },
                    'question': {
                        'type': 'string',
                        'description': 'Specific question about the image (optional)',
                        'default': 'What do you see in this image?'
                    },
                    'detail': {
                        'type': 'string',
                        'description': 'Level of detail in analysis',
                        'enum': ['low', 'high'],
                        'default': 'high'
                    }
                },
                'required': ['image_url']
            }
        }
    
    def execute(self, arguments: dict, context: AgentContext) -> str:
        """Analyze an image using GPT-4 Vision."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": arguments.get('question', 'What do you see in this image?')
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": arguments['image_url'],
                                    "detail": arguments.get('detail', 'high')
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            analysis = response.choices[0].message.content
            
            return json.dumps({
                'status': 'success',
                'image_url': arguments['image_url'],
                'analysis': analysis,
                'question': arguments.get('question', 'What do you see in this image?')
            })
            
        except Exception as e:
            return json.dumps({
                'status': 'error',
                'error': str(e),
                'message': 'Failed to analyze image.'
            })


class ImageVariationTool(ToolInterface):
    """
    Tool for creating variations of existing images.
    """
    
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def definition(self) -> dict:
        return {
            'name': 'create_variation',
            'description': 'Create a variation of an existing image',
            'parameters': {
                'type': 'object',
                'properties': {
                    'image_path': {
                        'type': 'string',
                        'description': 'Local path to the image file'
                    },
                    'n': {
                        'type': 'integer',
                        'description': 'Number of variations to create (1-10)',
                        'default': 1,
                        'minimum': 1,
                        'maximum': 10
                    },
                    'size': {
                        'type': 'string',
                        'description': 'Size of the variations',
                        'enum': ['256x256', '512x512', '1024x1024'],
                        'default': '1024x1024'
                    }
                },
                'required': ['image_path']
            }
        }
    
    def execute(self, arguments: dict, context: AgentContext) -> str:
        """Create variations of an image."""
        try:
            # Read the image file
            with open(arguments['image_path'], 'rb') as image_file:
                response = self.client.images.create_variation(
                    image=image_file,
                    n=arguments.get('n', 1),
                    size=arguments.get('size', '1024x1024')
                )
            
            variations = []
            for i, data in enumerate(response.data):
                variations.append({
                    'url': data.url,
                    'index': i
                })
            
            return json.dumps({
                'status': 'success',
                'original_image': arguments['image_path'],
                'variations': variations,
                'count': len(variations)
            })
            
        except Exception as e:
            return json.dumps({
                'status': 'error',
                'error': str(e),
                'message': 'Failed to create image variations.'
            })


class ProductionArtAgent(BaseAgent):
    """
    Production-ready AI Art Agent with real API integration.
    """
    name = 'production_art_agent'
    description = 'Professional AI art assistant with DALL-E and GPT-4 Vision'
    instructions_file = 'prompts/art_agent_instructions.md'  # You can create this file
    model = 'gpt-4o'
    tools = [
        ProductionImageGenerationTool,
        VisionAnalysisTool,
        ImageVariationTool
    ]
    
    def before_llm_call(self, messages: List[dict], tools: Optional[List[dict]]) -> None:
        """Track API usage."""
        # You could implement token counting, rate limiting, etc.
        print(f"🔄 Making LLM call with {len(messages)} messages...")
    
    def after_tool_result(self, tool_name: str, result: str, context: AgentContext) -> None:
        """Log successful image generations."""
        try:
            data = json.loads(result)
            if data.get('status') == 'success' and tool_name == 'generate_image':
                print(f"✨ Image generated successfully!")
                print(f"   URL: {data.get('image_url')}")
                print(f"   Revised prompt: {data.get('revised_prompt')}")
                
                # You could save URLs to a database, send notifications, etc.
                # self.save_to_gallery(data)
                
        except Exception as e:
            print(f"Error processing tool result: {e}")


# Example with OpenPipe configuration
class OpenPipeArtAgent(ProductionArtAgent):
    """
    Art agent configured for OpenPipe.
    """
    # OpenPipe allows model routing and optimization
    model = os.getenv('OPENPIPE_MODEL', 'gpt-4o')
    
    def __init__(self):
        # Configure for OpenPipe if needed
        if os.getenv('USE_OPENPIPE', 'false').lower() == 'true':
            # Override the OpenAI client configuration in tools
            # This is just an example - actual implementation depends on OpenPipe SDK
            pass


def create_art_agent_with_memory():
    """
    Example of creating an art agent with persistent memory/context.
    """
    context = AgentContext()
    
    # You could load previous conversation from a database
    # context.messages = load_from_database(user_id)
    
    # Add custom metadata
    context.metadata['user_preferences'] = {
        'preferred_style': 'vivid',
        'preferred_size': '1024x1024'
    }
    
    return ProductionArtAgent.with_context(context)


if __name__ == "__main__":
    # Make sure to set your API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)
    
    print("=== Production Art Agent Example ===\n")
    
    # Example: Generate an image
    agent = ProductionArtAgent()
    
    response = agent.run(
        "Create a breathtaking landscape of an alien planet with crystalline "
        "mountains and a purple sky with two suns setting on the horizon. "
        "Include bioluminescent plants in the foreground."
    )
    
    print(f"\nAgent Response: {response}")