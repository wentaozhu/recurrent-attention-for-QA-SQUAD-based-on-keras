import json

class Answer:
    def __init__(self, answer_start, text):
        self._answer_start_ = answer_start
        self._text_ = text

class QAEntity:
    def __init__(self, question, id):
        self._question_ = question
        self._id_ = id
        self._answers_ = []

class Paragraph:
    def __init__(self, context, qas):
        self._context_ = context
        self._qas_ = []
        for answer in qas:
            qa = QAEntity(answer['question'], answer['id'])
            for item in answer['answers']:
                a = Answer(item['answer_start'], item['text'])
                qa._answers_.append(a)
            self._qas_.append(qa)

class DataEntity:
    def __init__(self, title, paragraph_data):
        self._title_ = title
        self._paragraphs_ = []
        for item in paragraph_data:
            paragraph = Paragraph(item['context'], item['qas'])
            self._paragraphs_.append(paragraph)



def import_qas_data(datapath):
    with open(datapath) as data_file:
        data = json.load(data_file)

    data_entity_list = []
    for item in data['data']:
        entity = DataEntity(item['title'], item['paragraphs'])
        data_entity_list.append(entity)
    return data_entity_list

if __name__ == "__main__":
    path = '../../data/dev-v1.1.json'
    data_entity_list = import_qas_data(path)
    print(data_entity_list[0]._paragraphs_[0]._qas_[0]._id_)
    print(data_entity_list[0]._paragraphs_[0]._qas_[0]._question_)
    print(data_entity_list[0]._paragraphs_[0]._qas_[0]._answers_[0]._answer_start_)
    print(data_entity_list[0]._paragraphs_[0]._qas_[0]._answers_[0]._text_)
    print(len(data_entity_list[0]._paragraphs_[0]._qas_[1]._answers_))
    print(len(data_entity_list[0]._paragraphs_[0]._qas_))

    path = '../../data/train-v1.1.json'
    data_entity_list = import_qas_data(path)
    print(data_entity_list[0]._paragraphs_[0]._qas_[0]._id_)
    print(data_entity_list[0]._paragraphs_[0]._qas_[0]._question_)
    print(data_entity_list[0]._paragraphs_[0]._qas_[0]._answers_[0]._answer_start_)
    print(data_entity_list[0]._paragraphs_[0]._qas_[0]._answers_[0]._text_)
    print(len(data_entity_list[0]._paragraphs_[0]._qas_[1]._answers_))
    print(len(data_entity_list[0]._paragraphs_[0]._qas_))