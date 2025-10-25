import evaluate

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

def evaluate(reference, prediction):
    # ROUGE expects strings
    rouge_scores = rouge.compute(predictions=[prediction], references=[reference])

    # BLEU expects tokenized predictions and references
    bleu_scores = bleu.compute(
        predictions=[prediction.split()],
        references=[[reference.split()]]  # Note: nested list for multiple references
    )

    return {"ROUGE": rouge_scores, "BLEU": bleu_scores}

