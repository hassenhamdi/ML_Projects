# -*- coding: utf-8 -*-

!pip install git+https://github.com/huggingface/transformers

from transformers import AutoProcessor, AutoModelForVision2Seq
from huggingface_hub import hf_hub_download
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "ibm-granite/granite-vision-3.1-2b-preview"
processor = AutoProcessor.from_pretrained(model_path)

# load_in_4bit for 4bit quanyization
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model_8bit = AutoModelForVision2Seq.from_pretrained(
    "ibm-granite/granite-vision-3.1-2b-preview",
    quantization_config=quantization_config
)

from huggingface_hub import notebook_login

notebook_login()

model_8bit.push_to_hub("granite-vision-3.1-2b-preview-8bit")
