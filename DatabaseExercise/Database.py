from typing import List
from mysql import connector
from mysql.connector import errorcode


class Database(object):
    """
    Python Class for connecting  with MySQL server and accelerate development project using MySQL.
    """

    def __init__(self, **param):
        """
        This is the constructor create an object mysql_connector
        :param param: Parameter to connect data
        """
        super(Database, self).__init__()
        self.params = param
        self._connection = None
        self._cursor = None
        self.__create()

    def __create_connection(self):
        """
        This method used to create a connection to database using connector-python
        :return: the connection to database
        """
        if not self._connection:
            try:
                self._connection = connector.connect(**self.params)
                print("Connection successfully")
                return self._connection
            except connector.Error as err:
                if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                    print("Access denied! User or password is incorrect! Please check!")
                elif err.errno == errorcode.ER_BAD_DB_ERROR:
                    print("Database not found!")
                else:
                    print(err)

    def __get_cursor(self):
        """
        This method used to get cursor of the connection
        :return: cursor of the connection
        """
        if not self._cursor:
            if not self._connection:
                self.__create_connection()
            self._cursor = self._connection.cursor()
        return self._cursor

    def __create(self):
        """
        This method used to create connection and get the cursor
        """
        self.__create_connection()
        self.__get_cursor()

    def _commit(self):
        """
        This method used to commit the query to database
        """
        self._connection.commit()

    def execute(self, query: str, values: tuple = None):
        """
        This method used to execute query into database
        :param query: sql statement
        :param values: params to execute
        :return:
        """
        try:
            self._cursor.execute(query, values)
            if values:
                self._cursor.execute(query, values)
            else:
                self._cursor.execute(query)
        except (Exception, connector.Error) as err:
            print("Error occur when execute query into database:", err)

    def executemany(self, query: str, values: List[tuple]):
        """
        This method execute many query into database
        :param query: sql statement
        :param values: params to execute
        """
        try:
            self._cursor.executemany(query, values)
        except (Exception, connector.Error) as err:
            print("Error occur when execute query into database:", err)

    def fetch_all(self, query: str, need_fields: bool = False):
        """
        This method used to return values on a field in database
        :param query: sql statement
        :param need_fields: params to fetch
        :return:
        """
        self.execute(query=query)
        values = self._cursor.fetchall()
        if need_fields:
            field_names = [i[0] for i in self._cursor.description]
            return values, field_names
        return values

    def insert_one(self, query, value: tuple):
        """
        This method used to insert a new record in the table
        :param query: sql statement
        :param value: record to insert
        """
        self.execute(query=query, values=value)
        self._commit()

    def insert_list(self, query, values):
        """
        This method used to insert many records in the table
        :param query: sql statement
        :param values: list record to insert
        """
        self.executemany(query, values)
        self._commit()

    def delete(self, query):
        """
        This method used to delete the record in the table
        :param query: sql statement
        """
        self.execute(query)
        self._commit()

    def update(self, query):
        """
        This method used to update the record in the table
        :param query: sql statement
        """
        self.execute(query)
        self._commit()

    def select(self, query):
        """
        This method used to select a field in the table
        :param query: sql statement
        :return: List of records selected.
        """
        return self.fetch_all(query)

    def create_table(self, query):
        """
        This method used to create new table
        :param query: sql statement
        :return:
        """
        self.execute(query)


def main():
    database = Database(host="localhost", user="roots", password="BAPAiIntern2021@", port=3306, database="Test")
    # database.create_table("CREATE TABLE ABC(WORD_QUERY VARCHAR(25), MEANING VARCHAR(200)"
    #                       ", RELATED_WORDS VARCHAR(25), SIMILAR INT)")
    # sql = "INSERT INTO ABC(WORD_QUERY, MEANING, RELATED_WORDS, SIMILAR) VALUES(%s, %s, %s, %s)"
    # params = ('abcii', 'abc', 'def', 15)
    # database.insert_one(sql, params)
    sql1 = "SELECT * FROM ABC"
    data = database.select(sql1)
    print(data)


if __name__ == "__main__":
    main()
