import sys


def error_message_detail(error, error_details: sys):
    _, _, exc_tb = error_details.exc_info()
    if exc_tb is None:
        return str(error)

    file_name = exc_tb.tb_frame.f_code.co_filename
    return (
        "Error occurred in Python script name [{0}] line number [{1}] "
        "error message [{2}]"
    ).format(file_name, exc_tb.tb_lineno, str(error))


class CustomException(Exception):
    def __init__(self, error, error_details: sys):
        self.error_message = error_message_detail(error, error_details)
        super().__init__(self.error_message)

    def __str__(self):
        return self.error_message
