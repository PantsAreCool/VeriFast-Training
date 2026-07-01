# Task 2: Create a Multi-Image Comparison System
# Build a system that accepts a folder of product images and compares them across multiple dimensions. 
# For each image, the LLM should identify: product type, dominant color, visible features, estimated price range, and target audience. 
# Then produce a comparison table across all products. Use both OpenAI and Anthropic APIs and compare consistency of results. 
# Handle images of varying quality and resolution.

import os
import json
import base64
import glob
from openai import OpenAI

class MultiImageComparer:
    def __init__(self):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        
        self.models = {
            "openai": "openai/gpt-4o",
            "anthropic": "anthropic/claude-3.5-sonnet"
        }

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def analyze_product_image(self, image_path: str, model_provider: str) -> dict:
        base64_image = self._encode_image(image_path)
        model_id = self.models[model_provider]
        
        ext = os.path.splitext(image_path)[1].lower().replace(".", "")
        mime_type = f"image/{ext}" if ext in ["png", "gif", "webp"] else "image/jpeg"

        system_prompt = "You are a product catalog manager. Analyze the image and return a strict JSON object."
        user_prompt = """
        Analyze this product image across these specific dimensions:
        1. "product_type": general category (e.g., shoe, backpack, headphones).
        2. "dominant_color": primary visual color.
        3. "visible_features": summary string of key details or branding.
        4. "estimated_price_range": retail value estimation (e.g., "$50-$100").
        5. "target_audience": likely user base (e.g., "athletes", "students").

        Your response must match this structure exactly:
        {
            "product_type": "string",
            "dominant_color": "string",
            "visible_features": "string",
            "estimated_price_range": "string",
            "target_audience": "string"
        }
        """

        try:
            response = self.client.chat.completions.create(
                model=model_id,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": f"Failed to analyze with {model_provider}: {str(e)}"}

    def generate_comparison_report(self, folder_path: str):
        extensions = ('*.png', '*.jpg', '*.jpeg', '*.webp')
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))

        if not image_files:
            print(f"No images found in folder: {folder_path}")
            return

        
        for img_path in image_files:
            filename = os.path.basename(img_path)
            print(f"Product: {filename}")
            
            openai_res = self.analyze_product_image(img_path, "openai")
            anthropic_res = self.analyze_product_image(img_path, "anthropic")
            
            headers = ["Dimension", "OpenAI (GPT-4o)", "Anthropic (Claude 3.5)"]
            dimensions = ["product_type", "dominant_color", "visible_features", "estimated_price_range", "target_audience"]
            
            print(f"{headers[0]:<25} | {headers[1]:<30} | {headers[2]:<30}")
            print("-" * 90)
            for dim in dimensions:
                o_val = openai_res.get(dim, "N/A")
                a_val = anthropic_res.get(dim, "N/A")
                print(f"{dim:<25} | {str(o_val)[:30]:<30} | {str(a_val)[:30]:<30}")
            print("\n")

if __name__ == "__main__":
    
    comparer = MultiImageComparer()
    
    product_folder = "./Level 2 Module 1 Practical Tasks/06 - Multi-Modal LLMs/products" 
    
    if os.path.exists(product_folder):
        comparer.generate_comparison_report(product_folder)
    else:
        os.makedirs(product_folder, exist_ok=True)
        print(f"Created a sample folder at '{product_folder}'. Add images there to run the comparison system.")