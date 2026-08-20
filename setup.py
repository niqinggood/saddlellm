from pathlib import Path

from setuptools import setup, find_packages


ROOT = Path(__file__).resolve().parent
LONG_DESCRIPTION = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="saddlellm",
    version="2.32",
    author="niqinggood",
    author_email="niqinggood@163.com",
    license="MIT",
    description="LLM training, fine-tuning, distillation, quantization and deployment toolkit",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["__pycache__"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    # Static studio assets are declared explicitly below; disabling implicit
    # namespace scanning avoids treating web directories as Python packages.
    include_package_data=False,
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.6,<3.0",
        "transformers>=4.57.6,<5.0",
        "datasets>=3.6,<4.0",
        "peft>=0.18.1,<0.19",
        "trl>=0.26.2,<0.27",
        "bitsandbytes",
        "fastapi",
        "uvicorn",
        "python-multipart",
        "pydantic",
        "psutil",
        "nvidia-ml-py",
        "prometheus_client",
        "prometheus_fastapi_instrumentator",
        "evaluate",
        "numpy",
        "Pillow",
        "tqdm",
        "accelerate>=1.12,<2.0",
        "packaging>=23,<27",
        "pyyaml",
    ],
    extras_require={
        "posttrain": [
            "transformers==4.57.6",
            "datasets==3.6.0",
            "peft==0.18.1",
            "trl==0.26.2",
            "accelerate==1.12.0",
            "huggingface-hub==0.36.2",
            "tokenizers==0.22.2",
            "pyarrow==18.1.0",
            "packaging==26.2",
        ],
        "data": [
            "datasketch",
            "langdetect",
            "pyarrow",
            "simhash",
        ],
        "text": [
            "beautifulsoup4",
            "chardet",
            "dateparser",
            "emoji",
            "ftfy",
            "jieba",
            "nltk",
            "pandarallel",
            "pypinyin",
            "rapidfuzz",
            "scikit-learn",
            "sentence-transformers",
            "slimit",
            "unidecode",
            "zhconv",
            "zstandard",
        ],
    },
    entry_points={
        "console_scripts": [
            "saddle-llm=saddlellm.cli:main",
            "saddlellm=saddlellm.cli:main",
        ],
    },
    package_data={
        "saddlellm": [
            "spatial_studio_web/*",
            "spatial_studio_web/assets/*",
        ]
    },
)
