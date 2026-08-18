import os
from pathlib import Path

# Force the correct absolute path to saved_models so it works regardless of CWD
os.environ["SAVED_MODELS_DIR"] = str(Path(__file__).parent / "saved_models")

import gradio as gr
import uvicorn
from app.main import app as fastapi_app

# The Clever Trick:
# Hugging Face's "Gradio" space expects a Gradio interface at the root.
# We create a simple Gradio interface here so the HF Health Checks pass.
demo = gr.Blocks()
with demo:
    gr.Markdown("# MedSectAI API is Live")
    gr.Markdown("The frontend has been decoupled and deployed to Vercel.")
    gr.Markdown("This Hugging Face Space is now running purely as the Machine Learning API Backend.")

# We use Gradio's built-in function to mount this interface 
# onto YOUR existing FastAPI application at the root path ("/").
# Your FastAPI endpoints (/predict, /compare, /models) will still be accessible!
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

