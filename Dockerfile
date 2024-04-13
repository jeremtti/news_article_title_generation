FROM python:3.9

RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install torch>=1.3
RUN pip install fairseq
RUN pip install requests
RUN pip install scikit-learn 
RUN pip install tensorboardX
RUN pip install sentencepiece!=0.1.92
RUN pip install transformers 

RUN pip install accelerate>=0.12.0
RUN pip install datasets>=1.8.0

RUN pip install protobuf
RUN pip install rouge-score
RUN pip install nltk
RUN pip install py7zr 
RUN pip install evaluate
RUN pip install peft 


WORKDIR /app/codes 


CMD ["/bin/bash"]