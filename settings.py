import os
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_RAW_ROOT = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PREPROCESSED_ROOT = os.path.join(PROJECT_ROOT, 'data', 'pre_processed')
DATA_ENGINEER_ROOT = os.path.join(PROJECT_ROOT, 'data', 'engineered')
DATA_EXTERNAL_ROOT = os.path.join(PROJECT_ROOT, 'data', 'external')

MODELS_ROOT = os.path.join(PROJECT_ROOT, 'models')