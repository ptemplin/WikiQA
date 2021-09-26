from datasets import Dataset

from bert.preparedata import prepare_validation_features
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
)
from bert.util import postprocess_qa_predictions


PRETRAINED_MODEL_DIR = './save/'

QUESTION_COLUMN_NAME = "question"
CONTEXT_COLUMN_NAME = "context"
COLUMN_NAMES = ['id', 'context', 'question']


def load_model():
    # Load pre-trained model and tokenizer
    config = AutoConfig.from_pretrained(PRETRAINED_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_DIR, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(PRETRAINED_MODEL_DIR, config=config)

    return tokenizer, model


def post_processing_function(examples, features, predictions):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        version_2_with_negative=True,
        output_dir=PRETRAINED_MODEL_DIR
    )

    formatted_predictions = [
        {"id": k, "predictions": v} for k, v in predictions.items()
    ]
    return formatted_predictions


class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self, eval_dataset, eval_examples):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self.prediction_loop(eval_dataloader, description="Evaluation")

        # We might have removed columns from the dataset so we put them back.
        if isinstance(eval_dataset, Dataset):
            eval_dataset.set_format(type=eval_dataset.format["type"], columns=list(eval_dataset.features.keys()))

        eval_preds = post_processing_function(eval_examples, eval_dataset, output.predictions)

        return eval_preds


class ContextualQuestionAnswerer:

    def __init__(self):
        tokenizer, model = load_model()
        self.tokenizer = tokenizer
        self.model = model
        self.trainer = QuestionAnsweringTrainer(model=model, compute_metrics=None)

    def evaluate(self, question: str, context: str):
        """
        Extracts the answer to a factoid question from the necessary context. Will yield "no answer" if there is no
        answer of sufficiently high probability.

        :param question: to be answered about the passage contained in context
        :param context: containing the answer to the related question
        :return: list of ranked predictions (dicts) with 'text' and 'probability' keys
        """
        question_ds = Dataset.from_dict({
            'id': [1],
            'context': [context],
            'question': [question]
        })

        prepared_dataset = question_ds.map(
            lambda ds: prepare_validation_features(ds, self.tokenizer, QUESTION_COLUMN_NAME, CONTEXT_COLUMN_NAME),
            batched=True,
            remove_columns=COLUMN_NAMES,
            load_from_cache_file=True,
        )

        return self.trainer.evaluate(eval_dataset=prepared_dataset, eval_examples=question_ds)[0]['predictions']
