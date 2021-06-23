# https://tuna.thesaurus.com/relatedWords/
import requests
import Database

x = requests.get('https://tuna.thesaurus.com/relateWords/one')
db = db_test = {
        "host": "localhost",
        "user": "roots",
        "password": "BAPAiIntern2021@",
        "database": "SearchingSynonym",
        "port": "3306"
    }
