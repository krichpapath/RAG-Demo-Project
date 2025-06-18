import os
from mistralai import Mistral
from dotenv import load_dotenv
import base64
from pydantic import BaseModel, Field
from enum import Enum
from mistralai.extra import response_format_from_pydantic_model
from pypdf import PdfWriter, PdfReader
import time
import json
import asyncio

load_dotenv()

api_key = os.getenv('MISTRAL_API_KEY')


client = Mistral(api_key=api_key)


class ImageType(str, Enum):
    GRAPH = "graph"
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"

class Image(BaseModel):
    image_type: ImageType = Field(..., description="The type of the image. Must be one of 'graph', 'text', 'table' or 'image'.")
    description: str = Field(..., description="A description of the image.")

class Document(BaseModel):
    language: str = Field(..., description="The language of the document in ISO 639-1 code format (e.g., 'en', 'fr').")
    summary: str = Field(..., description="A summary of the document.")
    authors: list[str] = Field(..., description="A list of authors who contributed to the document.")

def encode_pdf(pdf_path):
    """Encode the pdf to base64."""
    try:
        with open(pdf_path, "rb") as pdf_file:
            return base64.b64encode(pdf_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {pdf_path} was not found.")
        return None
    except Exception as e:  # Added general exception handling
        print(f"Error: {e}")
        return None
    


async def process_chunk_async(start_page, end_page, reader, input_pdf_path):
    writer = PdfWriter()
    for i in range(start_page, end_page):
        writer.add_page(reader.pages[i])

    temp_name = f"chunk_{start_page+1}_to_{end_page}.pdf"
    temp_path = os.path.join(os.path.dirname(input_pdf_path), temp_name)

    with open(temp_path, "wb") as f:
        writer.write(f)

    base64_pdf = encode_pdf(temp_path)
    os.remove(temp_path)
    if not base64_pdf:
        return None

    def call_ocr():
        response = client.ocr.process(
            model="mistral-ocr-latest",
            pages=list(range(end_page - start_page)),
            document={"type": "document_url", "document_url": f"data:application/pdf;base64,{base64_pdf}"},
            bbox_annotation_format=response_format_from_pydantic_model(Image),
            document_annotation_format=response_format_from_pydantic_model(Document),
            include_image_base64=True
        )
        data = json.loads(response.model_dump_json())
        data["meta"] = {"start_page": start_page + 1, "end_page": end_page}
        return data

    return await asyncio.to_thread(call_ocr)

async def process_pdf_async(input_pdf_path, pages_per_chunk=8):
    reader = PdfReader(input_pdf_path)
    total = len(reader.pages)
    tasks = []

    for i in range(0, total, pages_per_chunk):
        end = min(i + pages_per_chunk, total)
        tasks.append(process_chunk_async(i, end, reader, input_pdf_path))

    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


def get_all_pdf(path_file):
    all_responses = asyncio.run(process_pdf_async(path_file))
    all_pdf = ""

    for response_dict in all_responses:
        for i in range(len(response_dict["pages"])):
            if response_dict["pages"][i]["images"] != []:
                for j in range(len(response_dict["pages"][i]["images"])):
                    image_caption = json.loads(response_dict["pages"][i]["images"][j]["image_annotation"])
                    all_pdf += image_caption["description"]

            all_pdf += response_dict["pages"][i]["markdown"]
    with open("./text_document/ocr_all_pdf.txt", "w", encoding="utf-8") as f:
        f.write(all_pdf)