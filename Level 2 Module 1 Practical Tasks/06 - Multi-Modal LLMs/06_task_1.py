# Task 1: Build a Screenshot Analyzer Tool
# Create a Python script that takes a screenshot image file as input and produces a structured analysis report. 
# The tool should: (1) detect if the screenshot contains code, a webpage, a chart, or a UI, 
# (2) extract any visible text with positioning information, (3) identify UI elements if it's an application screenshot, 
# and (4) generate accessibility-friendly alt text. Save the report as JSON. Test with at least 5 different types of screenshots.

import os
import json
import base64
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()


class ScreenshotAnalyzer:    
    def __init__(self):
        api_key=os.environ.get("MISTRAL_API_KEY")
        self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def analyze_screenshot(self, image_path: str, output_json_path: str = "analysis_report.json") -> dict:
        print(f"Analyzing {image_path}")
        base64_image = self._encode_image(image_path)
        
        system_prompt = (
            "You are an expert UI and visual design analysis tool. "
            "Analyze the provided screenshot and output your response strictly as a single JSON object."
        )

        user_prompt = """
        Please analyze this screenshot and extract the following information:
        1. "detected_type": Classify the primary content as one of ['code', 'webpage', 'chart', 'UI layout', 'mixed'].
        2. "extracted_text": List key strings of text along with their general layout positioning (e.g., 'top-left header', 'main body').
        3. "ui_elements": If a webpage or UI layout, list prominent UI elements detected (buttons, inputs, logos). If none, leave empty.
        4. "accessibility_alt_text": A concise, descriptive alternative text paragraph for visually impaired users.

        Provide the output in this exact JSON structure:
        {
            "detected_type": "string",
            "extracted_text": [{"text": "string", "position": "string"}],
            "ui_elements": ["string"],
            "accessibility_alt_text": "string"
        }
        """

        response = self.client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )

        report_data = json.loads(response.choices[0].message.content)
        
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4)
            
        print(f"Report successfully saved to {output_json_path}\n")
        return report_data

if __name__ == "__main__":

    api_key=os.environ.get("MISTRAL_API_KEY")
    analyzer = ScreenshotAnalyzer()
    
    try:
        sample_screenshot = "screenshot.png"
        
        if os.path.exists(sample_screenshot):
            report = analyzer.analyze_screenshot(sample_screenshot, "my_screenshot_report.json")
            print(json.dumps(report, indent=2))
        else:
            print(f"Please place a real screenshot at '{sample_screenshot}' to test the script.")
            
    except Exception as e:
        print(f"An error occurred: {e}")