from logger import log_standard_info
from psutil import virtual_memory

info_logger = log_standard_info('inference')

m1 = virtual_memory()
info_logger.info(f"The virtual memory info is: {m1}")
info_logger.info("Event seed table created")