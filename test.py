from model import Trigram

class Test:
    def __init__(self, model):
        self.model = model

    def test_read_model(self):
        self.model.read_model('data/model-br.en')
        print(self.model.trigram_model, len(self.model.trigram_model))

    def test_train(self):
        self.model.train()


model = Trigram(init_value=1, division_ratio=0.9)
test = Test(model)

test.test_train()
