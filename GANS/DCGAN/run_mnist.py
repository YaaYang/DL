# -*- encoding:utf-8 -*-
import os
import configparser

config = configparser.ConfigParser()
config_path = './model.conf'
config.read(config_path, encoding='utf-8')
DATA_DIR = config.get('PATHS', 'DATA_DIR')
OUTPUT_DIR = config.get('PATHS', 'OUTPUT_DIR')
SAVER_PATH = config.get('PATHS', 'SAVER_PATH')

python_shell = "python dc_gan_mnist.py " \
               "--DATA_DIR={DATA_DIR} --OUTPUT_DIR={OUTPUT_DIR} --SAVER_PATH={SAVER_PATH}". \
    format(DATA_DIR=DATA_DIR, OUTPUT_DIR=OUTPUT_DIR, SAVER_PATH=SAVER_PATH)

print(python_shell)
os.system(python_shell)
