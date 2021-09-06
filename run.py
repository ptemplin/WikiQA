import logging

from datasets import Dataset, DatasetDict

from bert.args import DataTrainingArguments
from bert.preparedata import prepare_validation_features
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
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

QUESTION = 'Who was Peter Badcoe?'


class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        # We might have removed columns from the dataset so we put them back.
        if isinstance(eval_dataset, Dataset):
            eval_dataset.set_format(type=eval_dataset.format["type"], columns=list(eval_dataset.features.keys()))

        eval_preds = self.post_process_function(eval_examples, eval_dataset, output.predictions)

        return eval_preds


def setup_logging():
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger.setLevel(logging.INFO)


def load_data():
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    return DatasetDict({'validation': Dataset.from_dict({
        'id': [1],
        'answers': [['']],
        'context': [CONTEXT],
        'question': [QUESTION],
        'title': ['']
    })})


def load_model():
    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(PRETRAINED_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_DIR, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(PRETRAINED_MODEL_DIR, config=config)

    return tokenizer, model


def run():
    setup_logging()

    # Load the data we're going to be using
    datasets = load_data()

    # Load the model and tokenizer
    print('Loading model...')
    tokenizer, model = load_model()
    print('Loaded model')

    question_column_name = "question"
    context_column_name = "context"
    answer_column_name = "answers"
    column_names = ['id', 'answers', 'context', 'question', 'title']

    validation_dataset = datasets["validation"].map(
        lambda dataset: prepare_validation_features(dataset,
                                                    tokenizer,
                                                    question_column_name,
                                                    context_column_name,
                                                    DataTrainingArguments()),
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=True,
    )

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
        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in datasets["validation"]]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    # Initialize our Trainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        eval_dataset=validation_dataset,
        eval_examples=datasets["validation"],
        post_process_function=post_processing_function,
        compute_metrics=lambda x: {},
    )

    results = trainer.evaluate()

    print("Question: %s" % QUESTION)
    print("Answer: %s" % results.predictions[0]['prediction_text'])


if __name__ == '__main__':
    run()
