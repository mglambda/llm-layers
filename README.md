# llm-layers

llm-layers determines suitable large language models for your hardware, downloads them from huggingface, and generates startup scripts for various backends that offload an appropriate number of layers onto your GPU.

The project got started because of my frustration with having to manage command line options to adjust the amount of layers that aN LLM backend (like llama.cpp) would load without going over my graphics card's vram. This was especially painful when loading several models at the same time, or having portions of vram be spoken for due to other reasons, like a TTS engine, whisper model, or streaming with OBS.

So at first this was just a csv file with model filename and number of GPU layers associated with that model. The `layers-file`. Now it looks something like this.

```
```