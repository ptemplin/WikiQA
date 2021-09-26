import logging

from bert.eval import ContextualQuestionAnswerer
from wiki.extract import extract_wiki


logger = logging.getLogger(__name__)

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

WIKI_PAGE = 'Four Candles'


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger.setLevel(logging.INFO)


def run():
    print('Loading model...')
    answerer = ContextualQuestionAnswerer()
    print('Loaded model')

    print('Downloading context...')
    context = extract_wiki(WIKI_PAGE)
    print('Context ready')

    question = input('What\'s your question?\n')
    while question:
        results = answerer.evaluate(question, context)
        print("I believe the answer is %s." % results[0]['text'])

        question = input('Any other questions?\n')


if __name__ == '__main__':
    setup_logging()
    run()
