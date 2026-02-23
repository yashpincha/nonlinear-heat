class BaseModel:
    def __init__(self, params):
        self.params = params

    @property
    def param_names(self):
        return list(self.params.keys())

    def forward(self, t, x):
        raise NotImplementedError
