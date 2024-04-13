# Description
This project (MS Data Science in Ecole Polytechnique, 2024) is joint work with Jeremie Touati to summarize French news articles and automatically find a title to these articles. 

We leverage pretrained models from HuggingFace. However, since our training data was in French and most models were English specific, we had to use multilingual or French specific transformers. For summarization such a good model is called Barthez.

Then we finetune this Barthez model on our data (file barthez_model_finetuning.py) using the HuggingFace Transformers library. 

Finally we use this finetuned model to generate titles (see file create_summaries_from_model.py). 

For more details see the report.pdf file in the report folder.



# Installation and usage 

Remark : since the finetuning is very memory consuming, it may not work on your personal computer. We recommend the use of servers (for instance on AWS) to make this code work (we use Ecole Polytechnique's servers).

## with Docker (recommended)

To build your Docker image the first time you can run from the root folder of this repository (Docker Desktop needs to be installed on your machine, this does not work on Mac computers)
```bash
docker build --tag news_article_title_generation_image .
```

Then at each use you need to run a container from your image, mounting your code and data as volumes so that they are available inside the container

```bash
docker run -it --name summarization-container -v C:/Users/pvanb/Projects/3A/inf582_nlp/data_challenge/codes:/app/codes -v C:/Users/pvanb/Projects/3A/inf582_nlp/data_challenge/data:/app/data news_article_title_generation_image /bin/bash
```

When you are done using your container go out of it 
```bash
exit
```
and then remove it 
```bash
docker rm summarization-container
```

## with a virtual environment 

The first time, you need to create your Python environment (Python 3.9 needs to be installed on your machine)
```bash
py -3.9 -m venv .venv
```

Then you need to activate it 
```bash
.venv\Scripts\activate
```
and install required packages 
```bash
py -m pip install -r requirements.txt
```

For subsequent uses you will just need to activate it as above, then run your codes, and then finally deactivate it.
```bash
deactivate
```







 

