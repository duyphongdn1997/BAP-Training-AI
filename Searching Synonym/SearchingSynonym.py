# https://tuna.thesaurus.com/relatedWords/
import requests
from Database import Database
import argparse


def main(opt):
    con = Database(host=opt.host, user=opt.user, password=opt.pass_word, port=opt.port, database=opt.database)
    if opt.crawl_data:
        if opt.create_table:
            con.create_table("CREATE TABLE SYNONYMS(WORD VARCHAR(35), MEANING VARCHAR(200), RELATED_WORD VARCHAR(35),"
                             " SIMILARITY INT, "
                             "PRIMARY KEY(WORD, RELATED_WORD))")
        with open("words.txt", "r+") as file:
            for word in file:
                word = word.strip()
                # print(stripped_line)
                req = requests.get('https://tuna.thesaurus.com/relatedWords/' + word)
                req = req.json()
                data = req["data"]
                if data is None:
                    continue
                else:
                    for entry in data:
                        # print("entry -->", type(entry))
                        definition = entry["definition"]
                        # print("definition -->", type(definition))
                        synonyms = entry["synonyms"]
                        for synonym in synonyms:
                            similarity = synonym["similarity"]
                            # print("similarity -->", type(similarity))
                            related_word = synonym["term"]
                            # print("related word -->", type(related_word))
                            sql = "INSERT INTO SYNONYMS(WORD, MEANING, RELATED_WORD, SIMILARITY) VALUES(%s, %s, %s, %s)"
                            value = (word, definition, related_word, similarity)
                            con.insert_one(sql, value)
    else:
        sql = 'SELECT * FROM SYNONYMS WHERE WORD="'+opt.get_data+'"'
        data = con.select(sql)
        for i in data:
            print(i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Choose mode')
    parser.add_argument('--crawl-data', action="store_true", help="Choose to crawl data.")
    parser.add_argument('--host', type=str, default="localhost", help="Host where we store database.")
    parser.add_argument('--port', type=str, default="3306", help="Port to store database.")
    parser.add_argument('--user', type=str, default="root", help="Username")
    parser.add_argument('--pass-word', type=str, default="", help="password")
    parser.add_argument('--database', type=str, default="", help="Name of the database.")
    parser.add_argument('--get-data', type=str, default="", help="Choose to get data from database.")
    parser.add_argument('--create-table', action="store_true", help="Create new table when crawl data.")
    opt = parser.parse_args()
    main(opt)


