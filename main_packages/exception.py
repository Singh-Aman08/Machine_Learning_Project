import sys
from logger import logger

def error_message_detail(error_msg, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno
    error_message = "Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name, line_no, str(error_msg)
    )
    return error_message

class CustomException(Exception):
    def __init__(self,error_mess,sys):
        super().__init__(error_mess)
        self.message = error_message_detail(error_mess, sys)
        
    def __str__(self):
        return self.message
    
    
if __name__ == "__main__":
    try:
        a = 1/0
    except Exception as e:
        print(e)
        logger.info(str(e))
        raise CustomException(e, sys)
        
        
        
        
    