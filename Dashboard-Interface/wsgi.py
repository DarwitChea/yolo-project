# wsgi.py

import os
from apps import create_app
from apps.config import config_dict

app_config = config_dict['Debug']  # or 'Production'
app = create_app(app_config)
