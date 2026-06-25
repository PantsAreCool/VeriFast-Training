# Task 3: Implement a Document Processing Pipeline
# Create a pipeline that processes PDF documents through multi-modal AI. 
# The pipeline should: (1) split a multi-page PDF into individual pages as images, (2) analyze each page for content type (text, table, chart, image), 
# (3) extract text from text-heavy pages, (4) describe charts and figures in detail, 
# (5) produce a consolidated summary of the full document. Track processing time and cost per page.

import os
import json
import base64
import time
from openai import OpenAI
import io
from pdf2image import convert_from_path

class PDFProcessingPipeline:
    def __init__(self):
        api_key = os.environ.get("MISTRAL_API_KEY")
        self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        self.model = "openai/gpt-4o"

    def _encode_image_bytes(self, image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def process_pdf(self, pdf_path: str):
        start_time = time.time()
        print(f"Starting Pipeline for: {os.path.basename(pdf_path)}")

        pages = convert_from_path(pdf_path, dpi=150)
        print(f"Successfully split PDF into {len(pages)} pages.\n")
        
        page_summaries = []
        pipeline_stats = []

        for i, page in enumerate(pages, start=1):
            page_start = time.time()
            base64_image = self._encode_image_bytes(page)
            
            user_prompt = """
            Analyze this document page image and output a strict JSON object with these fields:
            1. "content_type": Classify the primary layout style as one of ['text', 'table', 'chart', 'mixed'].
            2. "extracted_text": If 'text' or 'mixed', extract the primary written content exactly. If none, leave blank.
            3. "visual_description": If 'chart', 'table', or figures are present, describe their trends, values, and features in detail.
            4. "page_summary": A brief 2-sentence summary of what this page covers.

            JSON structure:
            {
                "content_type": "string",
                "extracted_text": "string",
                "visual_description": "string",
                "page_summary": "string"
            }
            """

            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                max_tokens=1000
            )

            result = json.loads(response.choices[0].message.content)
            page_summaries.append(result.get("page_summary", ""))
            
            latency = (time.time() - page_start) * 1000
            pipeline_stats.append({"page": i, "latency_ms": latency, "type": result.get("content_type")})
            
            print(f"[Page {i}] Type: {result.get('content_type')} | Time: {latency:.1f}ms")
            if result.get("visual_description"):
                print(f"   -> Visual Notes: {result.get('visual_description')[:100]}...")

        print("\nConsolidating full document summary...")
        summary_prompt = f"Based on these individual page summaries from a document, generate a final unified summary report:\n\n{chr(10).join(page_summaries)}"
        
        summary_response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=500
        )
        
        final_summary = summary_response.choices[0].message.content
        total_time = time.time() - start_time

        print("\n" + "="*50)
        print("FINAL PIPELINE REPORT")
        print("="*50)
        print(f"Total Processing Time: {total_time:.2f} seconds")
        print("\nPer-Page Breakdown:")
        for stat in pipeline_stats:
            print(f" - Page {stat['page']}: {stat['type']:<10} (Latency: {stat['latency_ms']:.0f}ms)")
        
        print("\nConsolidated Document Summary:")
        print(final_summary)
        print("="*50)

if __name__ == "__main__":
    pipeline = PDFProcessingPipeline()
    
    sample_pdf = "sample_report.pdf"
    
    if os.path.exists(sample_pdf):
        pipeline.process_pdf(sample_pdf)
    else:
        print(f"Please place a valid multi-page PDF document at '{sample_pdf}' to run the extraction pipeline.")