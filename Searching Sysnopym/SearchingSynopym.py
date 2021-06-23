# https://tuna.thesaurus.com/relatedWords/
import requests
from Database import Database


def main():

    con = Database(host="localhost", user="roots", password="BAPAiIntern2021@", port=3306, database="Test")
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


if __name__ == "__main__":
    main()
