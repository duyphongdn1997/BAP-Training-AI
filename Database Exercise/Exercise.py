from mysql import connector
from mysql.connector import errorcode


class DBHelper:
    """
    Python Class for connecting  with MySQL server and accelerate development project using MySQL.
    """
    DB_HOST = 'localhost'
    DB_PORT = 3306
    DB_USER = 'roots'
    DB_PASSWORD = 'BAPAiIntern2021@'
    DB_NAME = 'test'

    connected = False
    cursor = None
    connection = None
    insert_id = ""
    row_count = 0
    data = {}

    def __init__(self, host: str = DB_HOST, port: int = DB_PORT, user: str = DB_USER,
                 passwd: str = DB_PASSWORD, db: str = DB_NAME):
        """
        This is the constructor create an object mysql_connector
        :param host: The host name or IP address of the MySQL server.
        :param port: The TCP/IP port of the MySQL server. Must be an integer.
        :param user: The user name used to authenticate with the MySQL server.
        :param passwd: The password to authenticate the user with the MySQL server.
        :param db: The database name to use when connecting with the MySQL server.
        """
        self.__host = host
        self.__port = port
        self.__user = user
        self.__passwd = passwd
        self.__db = db
        if not self.connected:
            self.connect_mysql()

    def create_table(self, table, params=None):
        """
        This method create a new table.
        :param table: Name of the table
        :param params: params to execute sql statement
        """
        if params is None:
            params = {}

        # try:
        query_ = ""
        for key, value in params.items():
            query_ += key + " " + value + ', '
        self.connection = self.connect_mysql()
        query = "CREATE TABLE " + table + " (" + query_[:-2] + ", primary key (" + key + "))"
        self.cursor.execute(query)
        # except Exception:
        #     print("-----------CREATE TABLE ERROR-----------")
        #     print("SQL Error: {}".format(self.cursor.statement))

    def connect_mysql(self):
        """
        This is connect function
        :except: ER_ACCESS_DENIED_ERROR: raise print("User or password is incorrect!")
                 ER_BAD_DB_ERROR: raise print("Database not found!")
        """
        try:
            # if not self.connected:
            conn = connector.connect(host=self.__host,
                                     port=self.__port,
                                     user=self.__user,
                                     passwd=self.__passwd,
                                     db=self.__db,
                                     charset='utf8')
            self.cursor = conn.cursor(buffered=True)
            self.connected = True
            return conn
        except connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Access denied! User or password is incorrect! Please check!")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("Database not found!")
            else:
                print(err)

    def insert(self, table: str, data=None):
        """
        This method used to insert new records in the table
        :param table: Name of the table
        :param data: The records used to insert
        """
        if data is None:
            data = {}
        value_names = []
        values = []
        param_storage = []
        params = []
        try:
            if data is dict:
                for key, value in data.items():
                    value_names.append(key)
                    param_storage.append(str(value))
                    values.append('%s')
                params = tuple(param_storage)
                self.connection = self.connect_mysql()
                query = ("INSERT INTO " + table + "(" + ', '.join(value_names) +
                         ") ""VALUES(" + ', '.join(values) + ")")
                self.cursor.execute(query, params)
            elif isinstance(data, list):
                for data_ in data:
                    value_names = []
                    values = []
                    param_storage = []
                    print(data_)
                    for key, value in data_.items():
                        value_names.append(key)
                        param_storage.append(str(value))
                        values.append('%s')
                    params_ = tuple(param_storage)
                    params.append(params_)
                print(params)
                self.connection = self.connect_mysql()
                query = ("INSERT INTO " + table + "(" + ', '.join(value_names) +
                         ") ""VALUES(" + ', '.join(values) + ")")
                print(query)
                self.cursor.executemany(query, params)
            if hasattr(self.cursor, 'lastrowid'):
                self.insert_id = self.cursor.lastrowid
            self.row_count = self.cursor.rowcount
            self.connection.commit()

        except Exception:
            print("-----------INSERT ERROR-----------")
            print("SQL Error: {}".format(self.cursor.statement))

    def select(self, table, where=None, field="*"):
        """
        This method used to insert new records in the table
        :param table: Name of the table
        :param where: Sql condition
        :param field: list of field need to query
        """

        if where is None:
            where = {}
        where_or = ''
        where_and = ''
        where_params_storage = []
        where_params = []
        try:
            if len(where) > 0:
                if type(where) is dict:
                    # SELECT "AND" TYPE ##################################
                    x = 1
                    where_items_count = len(where)
                    for key in where:
                        if x == where_items_count:
                            where_and += (key + "='" + where[key] + "'")
                        else:
                            where_and += (key + "='" + where[key] + "' AND ")
                        x += 1
                    query = "SELECT " + field + " FROM " + table + " WHERE " + where_and + ""
                elif isinstance(where, list):
                    # SELECT "OR" TYPE ###################################
                    x = 1
                    where_items_count = len(where)
                    for where_ in where:
                        for key, value in where_.items():
                            if x == where_items_count:
                                where_or += (key + " = %s ")
                            else:
                                where_or += (key + " = %s OR ")

                            where_params_storage.append(str(value))
                            x += 1
                    where_params = tuple(where_params_storage)
                    query = "SELECT " + field + " FROM " + table + " WHERE " + where_or + ""
                else:
                    # SELECT "OTHER" TYPE ################################
                    query = "SELECT " + field + " FROM " + table + " WHERE " + where + ""

                self.connection = self.connect_mysql()
                self.cursor.execute(query, where_params)
            else:
                query = "SELECT " + field + " FROM " + table + ""
                self.connection = self.connect_mysql()
                self.cursor.execute(query)
            # return select output
            return self.cursor.fetchall()

        except Exception:
            print("-----------SELECT ERROR-----------")
            print("SQL Error: {}".format(self.cursor.statement))

    def update(self, table: str, parameters=None):
        """
        This method used to modify the existing records in the table.
        :param table: Name of the table
        :param parameters: new records to modify.
        """
        if parameters is None:
            parameters = {}
        value_name = []
        params = []
        try:
            for key, value in parameters.items():
                value_name.append(key + " = %s")
                params.append(value)

            query = "UPDATE " + table + " SET " + ', '.join(value_name) + ""
            params = tuple(params)
            self.connection = self.connect_mysql()
            self.cursor.execute(query, params)
            if hasattr(self.cursor, 'lastrowid'):
                self.insert_id = self.cursor.lastrowid
            self.row_count = self.cursor.rowcount
            self.connection.commit()

            return self.call_row_count()  # Return 1 or 0. If 1 is returned, the update has been completed; 0 is false.

        except Exception:
            print("-----------UPDATE ERROR-----------")
            print("SQL Error: {}".format(self.cursor.statement))

    def delete(self, table, parameters=None):
        """
        This method used to delete existing records in a table
        :param table: name of the table
        :param parameters: the records need to deleted
        :return:
        """
        # global values
        if parameters is None:
            parameters = {}
        params = []
        value_delete = []
        # print("DELETE")
        where = "WHERE " + str(parameters)
        query = "DELETE FROM " + table + " " + where + ""
        # print(params)
        params = tuple(params)
        self.connection = self.connect_mysql()
        self.cursor.execute(query, params)
        self.connection.commit()
        return self.call_row_count()

    def call_row_count(self):
        return self.row_count


if __name__ == '__main__':
    connect_ = DBHelper()

    testInsertParams = [{'names': 'ah4324met',
                         'fak': 'testF123ak',
                         'Id': '94'
                         },
                        {
                         'names': 'ahmet',
                         'fak': 'testFak',
                         'Id': '100'
                        }]

    testUpdateParams = {
        'names': '565',
        'fak': 'asd'
    }

    testSelectParams = [
        {
            'Id': '94'
        },
        {
            'Id': '100'

        }
    ]

    create_test = {
        "names": "varchar(50)",
        "fak": "varchar(50)",
        "Id": "int"
    }
    connect_.create_table('table1', create_test)
    # testSelect = connect_.select('table1', testSelectParams)
    # connect_.insert('table1', testInsertParams)
    # connect_.update('table1', testUpdateParams)
    # testDelete2 = connect_.delete('table1', 'Id=100')

