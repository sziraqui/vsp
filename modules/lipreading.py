class WordReader:

    def __init__(self):
        self.name = "WordReader"
        self.model = None
        raise NotImplementedError(self.name + " constructor is not fully implemented")
    

    def create_model(self,params):
        raise NotImplementedError(self.name + " create_model is not yet implemented")

    
    def train_model(self, trainParams):
        raise NotImplementedError(self.name + " train_model is not yet implemented")

    
    def test_model(self, testParams):
        raise NotImplementedError(self.name + " test_model is not yet implemented")


    def predict_word(self):
        raise NotImplementedError(self.name + " predict_word is not yet implemented")


    def __str__(self):
        return self.name + ":\nModel = " + self.model

class SentenceReader:

    def __init__(self):
        self.name = "SentenceReader"
        self.model = None
        raise NotImplementedError(self.name + " constructor is not fully implemented")
    

    def create_model(self,params):
        raise NotImplementedError(self.name + " create_model is not yet implemented")

    
    def train_model(self, trainParams):
        raise NotImplementedError(self.name + " train_model is not yet implemented")

    
    def test_model(self, testParams):
        raise NotImplementedError(self.name + " test_model is not yet implemented")


    def predict_word(self):
        raise NotImplementedError(self.name + " predict_word is not yet implemented")


    def __str__(self):
        return self.name + ":\nModel = " + self.model