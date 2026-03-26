# rootlogger.py

"""
Root logger module to configure logging for the ML project.
Creates a timestamped log file in the 'logs' directory and sets
up a consistent logging format for all modules.
"""

import logging
import os
from datetime import datetime
from typing import Any

# Generate timestamped log file name
LOG_FILE: str = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define logs directory (ensure it exists)
logs_path: str = os.path.join(
    r"C:\Users\Aman Kumar Singh\Desktop\ML_Project\Machine_Learning_Project", "logs"
)
os.makedirs(logs_path, exist_ok=True)

# Full path to the log file
LOG_FILE_PATH: str = os.path.join(logs_path, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Logger object to be imported and used in other modules
logger: Any = logging.getLogger()
logger.info(f"Logging initialized. Log file created at: {LOG_FILE_PATH}")

     

