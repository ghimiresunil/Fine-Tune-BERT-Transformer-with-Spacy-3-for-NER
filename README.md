# Fine-Tune-BERT-Transformer-with-Spacy-3-for-NER

![spacy](https://user-images.githubusercontent.com/40186859/150681117-53fdce05-e9d0-456c-b8a8-ef6feafc311a.jpg)

## Quick Intro
In this github repo, I will show how to train a BERT Transformer for Name Entity Recognition task using the latest Spacy 3 library. I am going to train an NER classifier to extract entities from scientific abstracts. 

Note: Fine tuning transformers requires a powerful GPU with parallel processing. For this I am using Google Colab since it provides freely available servers with GPUs.

<b> So, question may arise in your mind, What is BERT transformer? </b>

BERT stands for Bidirectional Encoder Representations from Transformers— leverages the transformer architecture in a novel way. For example, BERT analyses both sides of the sentence with a randomly masked word to make a prediction. In addition to predicting the masked token, BERT predicts the sequence of the sentences by adding a classification token [CLS] at the beginning of the first sentence and tries to predict if the second sentence follows the first one by adding a separation token[SEP] between the two sentences.

## Dataset or Data Labeling for Train and Test Set

_**Dataset Requirement**_
* Training and dev data in the spaCy 3 JSON Format 
  * Step 01: First, Generate IOB format in .tsv format using UBIAI Text Annotation Tool
  * Step 02: Convert to SpaCy JSON Format 

_**Note**:_ The training data and testing data was obtained using the **UBIAI** Text Annotation Tool.

You might get confused about **UBIAI**, following are the features about UBIAI Text Annotation Tool:

* ML auto-annotation
* Dictionary, regex, and rule-based auto-annotation
* Team collaboration to share annotation tasks
* Direct annotation export to IOB format

## Model Training

* Open a new Google Colab project and make sure to select GPU as hardware accelerator in the notebook settings.
* In order to accelerate the training process, we need to run parallel processing on our GPU. To this end we install the NVIDIA 9.2 cuda library:

Command 01
```bash
  !wget https://developer.nvidia.com/compute/cuda/9.2/Prod/local_installers/cuda-repo-ubuntu1604-9-2-local_9.2.88-1_amd64 -O cuda-repo-ubuntu1604–9–2-local_9.2.88–1_amd64.deb
```
Command 02
```bash
  !dpkg -i cuda-repo-ubuntu1604–9–2-local_9.2.88–1_amd64.deb
```
Command 03
```bash
  !apt-key add /var/cuda-repo-9–2-local/7fa2af80.pub
```
Command 04
```bash
  !apt-get update
```
Command 05
```bash
  !apt-get install cuda-9.2
```
To check the correct cuda compiler is installed, run: !nvcc --version

* Install the spacy library and spacy transformer pipeline:

```bash
  pip install -U spacy 
  !python -m spacy download en_core_web_trf
```
* Next, we install the pytorch machine learning library that is configured for cuda 9.2:

```bash
  pip install torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

* After pytorch install, we need to install spacy transformers tuned for cuda 9.2 and change the CUDA_PATH and LD_LIBRARY_PATH as below. Finally, install the cupy library which is the equivalent of numpy library but for GPU:

```bash
!pip install -U spacy[cuda92,transformers]
!export CUDA_PATH=”/usr/local/cuda-9.2"
!export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
!pip install cupy
```
* SpaCy 3 uses a config file config.cfg that contains all the model training components to train the model. In spaCy training page, you can select the language of the model (English in this tutorial), the component (NER) and hardware (GPU) to use and download the config file template

```bash
   # This is an auto-generated partial config. To use it with 'spacy train'
   # you can run spacy init fill-config to auto-fill all default settings:
   # python -m spacy init fill-config ./base_config.cfg ./config.cfg
   [paths]
   train = "drive/MyDrive/NER_data/train.spacy"
   dev = "drive/MyDrive/NER_data/test.spacy"

   [system]
   gpu_allocator = "pytorch"

   [nlp]
   lang = "en"
   pipeline = ["transformer","ner"]
   batch_size = 128

   [components]

   [components.transformer]
   factory = "transformer"

   [components.transformer.model]
   @architectures = "spacy-transformers.TransformerModel.v1"
   name = "roberta-base"
   tokenizer_config = {"use_fast": true}

   [components.transformer.model.get_spans]
   @span_getters = "spacy-transformers.strided_spans.v1"
   window = 128
   stride = 96

   [components.ner]
   factory = "ner"

   [components.ner.model]
   @architectures = "spacy.TransitionBasedParser.v2"
   state_type = "ner"
   extra_state_tokens = false
   hidden_width = 64
   maxout_pieces = 2
   use_upper = false
   nO = null

   [components.ner.model.tok2vec]
   @architectures = "spacy-transformers.TransformerListener.v1"
   grad_factor = 1.0

   [components.ner.model.tok2vec.pooling]
   @layers = "reduce_mean.v1"

   [corpora]

   [corpora.train]
   @readers = "spacy.Corpus.v1"
   path = ${paths.train}
   max_length = 500

   [corpora.dev]
   @readers = "spacy.Corpus.v1"
   path = ${paths.dev}
   max_length = 0

   [training]
   accumulate_gradient = 3
   dev_corpus = "corpora.dev"
   train_corpus = "corpora.train"

   [training.optimizer]
   @optimizers = "Adam.v1"

   [training.optimizer.learn_rate]
   @schedules = "warmup_linear.v1"
   warmup_steps = 250
   total_steps = 20000
   initial_rate = 5e-5

   [training.batcher]
   @batchers = "spacy.batch_by_padded.v1"
   discard_oversize = true
   size = 2000
   buffer = 256

   [initialize]
   vectors = null
```
_**Note: fill out the path for the train and dev .spacy files. Once done, we upload the file to Google Colab.**_

* Now we need to auto-fill the config file with the rest of the parameters that the BERT model will need; all you have to do is run this command:

```bash
   !python -m spacy init fill-config ./config_folder/config.cfg ./config_folder/config_train_spacy.cfg
```

_**Note: I suggest to debug your config file in case there is an error:**_

```bash
   !python -m spacy debug data drive/MyDrive/config.cfg
```

* Finally ready to train the BERT model! Just run this command and the training should start

```bash
   !python -m spacy train -g 0 ./config_folder/config_train_spacy.cfg — output ./output_folder/
```

_**P.S: if you get the error cupy_backends.cuda.api.driver.CUDADriverError: CUDA_ERROR_INVALID_PTX: a PTX JIT compilation failed, just uninstall cupy and install it again and it should fix the issue.**_

* If everything went correctly, the model will generate the model scores and losses  

### !!! Hooray, the model will be saved under folder model-best. The model scores are located in meta.json file inside the model-best folder !!!

## Inference NER
- Clone this repository on your local machine:
```
https://github.com/ghimiresunil/Fine-Tune-BERT-Transformer-with-Spacy-3-for-NER.git
```
- Create a virtual environment
```
python -m venv venv
source venv/bin/activate  # On Linux/Mac
venv\Scripts\activate  # On Windows
```
- Install the required dependencies:
```
pip install -r requirements.txt
```
- Mail to `info@sunilghimire.com.np` for Fine tune NER Model
- Run the program
```
uvicorn api:app --reload
```

## Build docker image
- Build the Docker image using the updated Dockerfile:
```
docker build -t my-fastapi-app .
```
Note: This command builds the Docker image with the tag `my-fastapi-app`.
- Run the Docker container:
```
docker run -it --rm -p 80:80 my-fastapi-app bash
```

## Final Output
```
{
  "parsed_output": {
    "Name": "Sunil Ghimire",
    "Phone": "+977 9841070311",
    "EMAIL": "info@sunilghimire.com.np",
    "Hard Skills": [
      "python",
      "java",
      "c",
      "dsa",
      "oops",
      "html",
      "pandas",
      "numpy",
      "scikit",
      "keras",
      "tensorflow",
      "flask",
      "git",
      "heroku",
      "git",
      "flask",
      "heroku",
      "python",
      "cnn",
      "clahe",
      "transfer learning",
      "opencv",
      "clahe",
      "bm3d",
      "xception",
      "resnet50v2",
      "cnn",
      "rnn",
      "lstm",
      "cnn",
      "opencv",
      "data visualization",
      "cctv",
      "rnn",
      "lstm",
      "inceptionv3",
      "lof",
      "correlation map",
      "smote",
      "random forest classifier",
      "inter-quartile range (iqr)",
      "smote",
      "correlation map",
      "random forest classifier"
    ]
  }
}
```
