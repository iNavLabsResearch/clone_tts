"""milli_tts: finetune SPRINGLab/Indic-Mio (MioTTS) on Gujarati Vaani data.

The model is a Qwen3ForCausalLM that emits MioCodec audio tokens (``<|s_N|>``).
This package provides the data-prep, training, and inference building blocks.
"""

__version__ = "0.1.0"
