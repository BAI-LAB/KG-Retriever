import jieba
import evaluate
from text2vec import Similarity

def bleu_score(
        continuation: list,
        reference: list,
        with_penalty=False
) -> float:
    f = lambda text: list(jieba.cut(text))
    bleu = evaluate.load('./src/metric/.cache/huggingface/bleu')
    results = bleu.compute(predictions=continuation, references=reference, tokenizer=f)

    bleu_avg = results['bleu']
    bleu1 = results['precisions'][0]
    bleu2 = results['precisions'][1]
    bleu3 = results['precisions'][2]
    bleu4 = results['precisions'][3]
    brevity_penalty = results['brevity_penalty']

    if with_penalty:
        return bleu_avg, bleu1, bleu2, bleu3, bleu4
    else:
        return 0.0 if brevity_penalty == 0 else bleu_avg / brevity_penalty, bleu1, bleu2, bleu3, bleu4

def rougeL_score(
    continuation: list,
    reference: list
) -> float:
    f = lambda text: list(jieba.cut(text))
    rouge = evaluate.load('./src/metric/.cache/huggingface/rouge')
    results = rouge.compute(predictions=continuation, references=reference, tokenizer=f, rouge_types=['rougeL'])
    score = results['rougeL']
    return score

def bert_score(
    continuation: list,
    reference: list
) -> float:
    """
    Note:
        Requesting the network to connect to Hugging Face.
    """
    scores = []
    n = len(continuation)
    sim = Similarity(model_name_or_path="./src/metric/.cache/text2vec-base-chinese")
    for i in range(n):
        score = sim.get_score(continuation[i], reference[i])
        scores.append(score)
    return sum(scores)/n