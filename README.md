# llm-layers

llm-layers determines suitable large language models for your hardware, downloads them from huggingface, and generates startup scripts for various backends that offload an appropriate number of layers onto your GPU.

The project got started because of my frustration with having to manage command line options to adjust the amount of layers that aN LLM backend (like llama.cpp) would load without going over my graphics card's vram. This was especially painful when loading several models at the same time, or having portions of vram be spoken for due to other reasons, like a TTS engine, whisper model, or streaming with OBS.

So at first this was just a csv file with model filename and number of GPU layers associated with that model. The `layers-file`. Now it looks something like this.

```
 $ llm-layers -d
# Running with -d (--dry_run), Nothing permanent will be written to disk. Here isa pretty version of the potential layers file.
# Run with -g to actually generate the layers file and the scripts.

name                                               gpu_layers    context  prompt_format           type
-----------------------------------------------  ------------  ---------  ----------------------  ----------
Noromaid-v0.4-Mixtral-Instruct-8x7b.q4_k_m.gguf            24       4096  chat-ml                 chat
daringmaid-20b.Q6_K.gguf                                   54       4096  alpaca                  chat
dolphin-2.1-mistral-7b.Q4_K_M.gguf                        999       8192  chat-ml                 default
dolphin-2.1-mistral-7b.Q8_0.gguf                          999       8192  chat-ml                 default
dolphin-2.7-mixtral-8x7b.Q4_0.gguf                         22       4096  chat-ml                 default
dolphin-2.7-mixtral-8x7b.Q4_K_M.gguf                       22       4096  chat-ml                 default
llava-v1.6-34b.Q4_K_M.gguf                                 54       2048  chat-ml                 multimodal
llava-v1.6-mistral-7b.Q5_K_M.gguf                         999       4096  chat-ml                 multimodal
miqu-1-70b.q5_K_M.gguf                                     32        512  mistral                 default
mistral-7b-instruct-v0.2.Q4_K_M.gguf                      999       8192  mistral                 default
neuralhermes-2.5-mistral-7b.Q6_K.gguf                       1       2048  chat-ml                 default
solar-10.7b-instruct-v1.0.Q8_0.gguf                       999       4096  user-assistant-newline  mini
unholy-v2-13b.Q8_0.gguf                                   666       8192  alpaca                  chat
```

# Usage

Coming soon.


