class Sun_Model:
    def __init__(self, fin_model, summary, aug_models, MAE, train_y, train_x, test_y, test_x, y_pred):
        self.fin_model = fin_model
        self.summary = summary
        self.MAE = MAE
        self.aug_models = aug_models
        self.train_y = train_y
        self.train_x = train_x
        self.test_y = test_y
        self.test_x = test_x
        self.y_pred = y_pred
