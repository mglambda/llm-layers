from setuptools import setup, find_packages
import os


LLM_LAYERS_VERSION='0.2.0'


with open("README.md", "r", encoding="utf-8") as readme_file:
    README = readme_file.read()

setup(
    name='llm-layers',
    version=LLM_LAYERS_VERSION,
    url="https://github.com/mglambda/llm-layers",
    author="Marius Gerdes",
    author_email="integr@gmail.com",
    description="llm-layers determines suitable large language models for your hardware, downloads them from huggingface, and generates startup-scripts that offload an appropriate number of layers onto your GPU.",
    long_description=README,
    long_description_content_type="text/markdown",
    license_files=["LICENSE"],
    scripts=["scripts/llm-layers"],
    packages=find_packages(include=['llm_layers']),
    install_requires=["appdirs"]
)

