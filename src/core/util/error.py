class ErrorHandler(Exception):
    def __init__(self, message, code=400):
        self.message = message
        self.code = code
        super().__init__(f"[Error {code}]: {message}")