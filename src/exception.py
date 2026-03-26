# exception.py

"""
Custom exception module for handling and logging errors.
Provides detailed information about the exception including
file name, line number, and error message.
"""

import sys
from rootlogger import logger


def error_message_detail(error_msg: str, error_detail: sys) -> str:
    """
    Generates a detailed error message including the file name,
    line number, and the original error message.

    Args:
        error_msg (str): The original error message.
        error_detail (sys): The sys module for fetching exception info.

    Returns:
        str: Formatted error message with file and line number.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name: str = exc_tb.tb_frame.f_code.co_filename
    line_no: int = exc_tb.tb_lineno
    error_message: str = (
        f"Error occurred in python script name [{file_name}] "
        f"line number [{line_no}] error message [{error_msg}]"
    )
    return error_message


class CustomException(Exception):
    """
    Custom exception class that wraps standard exceptions to include
    detailed debugging information (file, line number, error message).
    """

    def __init__(self, error_mess: str, sys_module: sys):
        """
        Initializes the CustomException instance.

        Args:
            error_mess (str): The original error message.
            sys_module (sys): The sys module for exception info.
        """
        super().__init__(error_mess)
        self.message: str = error_message_detail(error_mess, sys_module)
        logger.error(self.message)

    def __str__(self) -> str:
        """
        Returns the detailed error message when printed.

        Returns:
            str: Detailed error message.
        """
        return self.message
    
    

        
        
        
    