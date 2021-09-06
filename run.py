import logging

from datasets import Dataset

from bert.args import DataTrainingArguments
from bert.preparedata import prepare_validation_features
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
)
from bert.util import postprocess_qa_predictions


logger = logging.getLogger(__name__)


PRETRAINED_MODEL_DIR = './save/'

DATA_PARAMS = DataTrainingArguments()

CONTEXT = """
Peter Badcoe (11 January 1934 – 7 April 1967) was an Australian recipient of the Victoria Cross, 
the highest award for gallantry in battle that could be awarded at that time to a member of 
the Australian armed forces. Badcoe joined the Australian Army in 1950 and graduated from the 
Officer Cadet School, Portsea, in 1952. Posted to South Vietnam in 1966, Badcoe displayed conspicuous 
gallantry and leadership on three occasions between February and April 1967. In the final battle, 
he was killed by a burst of machine-gun fire. He was posthumously awarded the Victoria Cross 
for his actions, as well as the United States Silver Star and several South Vietnamese medals. 
Badcoe\'s medal set is now displayed in the Hall of Valour at the Australian War Memorial 
in Canberra. Buildings in South Vietnam and Australia have been named after him, as has a 
perpetual medal at an Australian Football League match held on Anzac Day."""

QUESTION_COLUMN_NAME = "question"
CONTEXT_COLUMN_NAME = "context"
COLUMN_NAMES = ['id', 'context', 'question']


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger.setLevel(logging.INFO)


# Post-processing:
def post_processing_function(examples, features, predictions):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        version_2_with_negative=True,
        output_dir='./save/'
    )

    formatted_predictions = [
        {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
    ]
    return formatted_predictions


class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.post_process_function = post_process_function

    def evaluate(self, eval_dataset, eval_examples):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self.prediction_loop(eval_dataloader, description="Evaluation")

        # We might have removed columns from the dataset so we put them back.
        if isinstance(eval_dataset, Dataset):
            eval_dataset.set_format(type=eval_dataset.format["type"], columns=list(eval_dataset.features.keys()))

        eval_preds = self.post_process_function(eval_examples, eval_dataset, output.predictions)

        return eval_preds


def load_model():
    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(PRETRAINED_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_DIR, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(PRETRAINED_MODEL_DIR, config=config)

    return tokenizer, model


def run():
    # Load the model and tokenizer
    print('Loading model...')
    tokenizer, model = load_model()
    print('Loaded model')

    trainer = QuestionAnsweringTrainer(
        model=model,
        post_process_function=post_processing_function,
        compute_metrics=None,
    )

    question = input('What\'s your question?\n')
    while question:
        question_ds = Dataset.from_dict({
            'id': [1],
            'context': [CONTEXT],
            'question': [question]
        })

        prepared_dataset = question_ds.map(
            lambda ds: prepare_validation_features(ds,
                                                   tokenizer,
                                                   QUESTION_COLUMN_NAME,
                                                   CONTEXT_COLUMN_NAME,
                                                   DataTrainingArguments()),
            batched=True,
            remove_columns=COLUMN_NAMES,
            load_from_cache_file=True,
        )

        predictions = trainer.evaluate(eval_dataset=prepared_dataset, eval_examples=question_ds)
        print("I believe the answer is %s." % predictions[0]['prediction_text'])

        question = input('Any other questions?\n')


if __name__ == '__main__':
    setup_logging()
    run()
