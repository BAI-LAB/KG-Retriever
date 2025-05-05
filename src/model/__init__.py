from src.model.graphrag_llm import GraphRAGLLM

load_model = {
    'graphrag_llm': GraphRAGLLM,
}

llm_model_path = {
    'baichuan_13b_chat': 'Baichuan-13B-Chat',
    'qwen_7b_chat': 'Qwen1.5-7B-Chat',
    'qwen_7b': 'Qwen1.5-7B',
    'qwen_14b': 'Qwen1.5-14B',
    'qwen_32b_chat': 'Qwen1.5-32B-Chat-AWQ',
    'llama2_7b': 'Llama-2-7b-hf',
    'glm_6b':'chatglm3-6b',
    'llama3_8b':'Meta-Llama-3-8B'
}