# Helpers
Helpers directory is standard code you can use in every project.

The two files include, config.py and logger.py

# Logger.py
When you import logger.py in your code, it will return a logging.Logger object. You will be able to use this to log your information to a log file and to screen. E.g.
```python
from helpers.logger import logger

logger.info('This is an INFO log')
```
```bash
$ pipenv run python app.py

[2022-04-22 20:14:15] [MainThread] {/apps/python-project-template/src/app.py:20} INFO - This is an INFO log
```

# Config.py
When you import config.py in you code, it will return a dictionary containing the information from your config file (.yaml, .toml or .json).
You need to set the following:
```bash
$ export PYTHON_ENV=name_of_config_file.extension

$ export PYTHON_CONFIG_DIR=path/to/config/directory
```