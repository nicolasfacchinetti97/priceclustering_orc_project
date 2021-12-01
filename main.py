from instance import instance
import configparser

# read configuration file
config = configparser.ConfigParser()
config.read("config.ini")
instance_path = config['inputfile']['fileName']

items = instance(instance_path)
