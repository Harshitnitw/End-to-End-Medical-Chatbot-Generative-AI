# End-to-end-Medical-Chatbot-Generative-AI


# How to run?
### STEPS:

Clone the repository

```bash
git clone https://github.com/Harshitnitw/End-to-End-Medical-Chatbot-Generative-AI
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n venv python=3.10 -y
```

```bash
conda activate venv
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone & MistralAI credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
MISTRAL_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


```bash
# run the following command to store embeddings to pinecone
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

### Now, open the URL to use your medical chatbot


### Techstack Used:

- Python
- LangChain
- Flask
- GPT
- Pinecone
