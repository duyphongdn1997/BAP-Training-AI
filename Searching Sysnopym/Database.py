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
        self._init_()

    def __create_connection(self):
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
        if not self._cursor:
            if not self._connection:
                self.__create_connection()
            self._cursor = self._connection.cursor()
        return self._cursor

    def __close(self):
        if self._connection:
            if self._cursor:
                self._cursor.close()
            self._connection.close()
            print("MySQL connection is closed!")
        self._cursor = None
        self._connection = None

    def _init_(self):
        self.__create_connection()
        self.__get_cursor()

    def _commit(self):
        self._connection.commit()

    def execute(self, query: str, values: tuple = None):
        """
        This method execute query into database
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

    def executemany(self, query: str, values: [tuple]):
        """

        :param query:
        :param values:
        :return:
        """
        try:
            self._cursor.executemany(query, values)
        except (Exception, connector.Error) as err:
            print("Error occur when execute query into database:", err)

    def fetch_one(self, query: str, need_fields: bool = False):
        """

        :param query:
        :param need_fields:
        :return:
        """
        self.execute(query=query)
        value = self._cursor.fetchall()
        if need_fields:
            field_names = [i[0] for i in self._cursor.description]
            return value, field_names
        return value

    def fetch_all(self, query: str, need_fields: bool = False):
        """

        :param query:
        :param need_fields:
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

        :param query:
        :param value:
        :return:
        """
        self.execute(query=query, values=value)
        self._commit()

    def insert_list(self, query, values):
        """

        :param query:
        :param values:
        :return:
        """
        self.executemany(query, values)
        self._commit()

    def delete_list(self, query):
        """

        :param query:
        :return:
        """
        self.execute(query)
        self._commit()

    def update_one(self, query):
        """

        :param query:
        :return:
        """
        self.execute(query)
        self._commit()

    def select(self, query):
        """

        :param query:
        :return:
        """
        self.execute(query)
        return self.fetch_all(query)

    def create_table(self, query):
        """

        :param query:
        :return:
        """
        self.execute(query)
