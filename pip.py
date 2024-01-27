from flask import Flask, render_template, request
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain

app = Flask(__name__)

repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
huggingfacehub_api_token = "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"
llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token,
                     repo_id=repo_id,
                     model_kwargs={"temperature":0.7, "max_new_tokens":500})

template = """Question: {question}
              prompt:you helpful."""

prompt = PromptTemplate(template=template, input_variables=["question"])

chain = LLMChain(prompt=prompt, llm=llm)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    question = request.form['question']
    out = chain.run(question)
    return str(out)

if __name__ == '__main__':
    app.run(debug=True)