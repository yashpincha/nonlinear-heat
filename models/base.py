class BaseModel:
    def __init__(self, params):
        """
        params: dict with model-specific parameter names and initial guesses
        e.g., {'diffusivity': 0.01, 'heat_loss': 0.1}
        """
        self.params = params

    @property
    def param_names(self):
        # returns list of params
        return list(self.params.keys())

    def forward(self, t, x):
        raise NotImplementedError
