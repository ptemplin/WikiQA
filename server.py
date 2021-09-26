from flask import Flask
from flask_restful import Resource, Api, reqparse

from bert.eval import ContextualQuestionAnswerer
from wiki.extract import extract_wiki

RESPONSE_HEADERS = {'Access-Control-Allow-Origin': "*"}


app = Flask(__name__)
api = Api(app)
answerer = ContextualQuestionAnswerer()

parser = reqparse.RequestParser()
parser.add_argument('question', required=True)
parser.add_argument('page', required=True)


class QAResource(Resource):

    def get(self):
        args = parser.parse_args()
        question = args['question']
        page = args['page']

        context, url = extract_wiki(page)
        answer = answerer.evaluate(question, context)[0]['text']
        start_index = max(context.find(answer) - 100, 0)
        end_index = min(context.find(answer) + len(answer) + 100, len(context))
        return {
                   'answer': answer,
                   'context': context[start_index:end_index],
                   'url': url
               }, 200, RESPONSE_HEADERS


api.add_resource(QAResource, '/')

if __name__ == '__main__':
    app.run(debug=True)
