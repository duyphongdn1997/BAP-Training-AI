from mysql import connector


class DBHelper:
    """
    This is a Database helper class
    """

    def __init__(self, host: str, port: int, user: str, passwd: str, db: str):
        """
        This is the constructor creat an object mysql_connector
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


    def connect_mysql(self):
        """
        This method sets up a connection but do not specify the database name
        """
        conn = connector.connect(host=self.__host,
                                 port=self.__port,
                                 user=self.__user,
                                 passwd=self.__passwd,
                                 # db=self.__db, #Do not specify the database name
                                 charset='utf8')
        return conn

    def connect_database(self):
        """
        This method sets up a connection
        """
        conn = connector.connect(host=self.__host,
                                 user=self.__user,
                                 password=self.__passwd,
                                 port=self.__port,
                                 database=self.__db,
                                 charset='utf8')
        return conn

    def create_database(self):
        """
        This method create database name self.__db
        """
        conn = self.connect_mysql()
        sql = 'create database if not exists ' + self.__db
        cur = conn.cursor()
        cur.execute(sql)
        cur.close()
        conn.close()

    def create_table(self, sql):
        """
        This method create a table using sql statement
        :param sql: sql statement
        """
        conn = self.connect_database()
        cur = conn.cursor()
        cur.execute(sql)
        cur.close()
        conn.close()

    def insert(self, sql, *params):
        """
        This method insert into table using sql statement
        :param sql: sql statement
        """
        conn = self.connect_database()
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
        cur.close()
        conn.close()

    def update(self, sql, *params):
        """
        This method update using sql statement
        :param sql: sql statement
        :param params: config for sql statement
        """
        conn = self.connect_database()
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
        cur.close()
        conn.close()

    def close(self, sql, *param):
        """
        This method close connector
        """
        conn = self.connect_database()
        conn.close()

    def delete(self, sql, *params):
        conn = self.connect_database()
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
        cur.close()
        conn.close()


class Person:
    def __init__(self, name: str, age: int, address: str):
        self.__name = name
        self.__age = age
        self.__address = address

    def get_name(self):
        return self.__name

    def get_age(self):
        return self.__age

    def get_address(self):
        return self.__address


if __name__ == '__main__':
    person0 = Person(name='a', age=1, address='10 abc')
    person1 = Person(name='b', age=2, address='11 def')
    person2 = Person(name='c', age=3, address='01 ghi')

    # connect SQL
    test_db_helper = DBHelper('localhost', '3306', 'roots', 'BAPAiIntern2021@', 'Test')
    test_db_helper.connect_mysql()  # connect SQL
    test_db_helper.create_database()  # create Database Test
    test_db_helper.connect_database()  # connect database Test

    # # create Table Person
    # sql = 'create table Person(name nvarchar(50) primary key, age int, address nvarchar(20))'
    # test_db_helper.create_table(sql)

    # # insert data
    sql = 'insert into Person(name, age, Address) values(%s,%s,%s,%s)'
    test_db_helper.insert(sql, (person0.get_name(), person0.get_age(), person0.get_address()))
    #
    # # update data
    # sql = 'update Person set age = %s, address = %s where name = %s'
    # params = (person1.get_age(), person1.get_address(), person1.get_name())
    # test_db_helper.update(sql, *params)
    #
    # # delete data
    # sql = 'delete from Person where name = %s'
    # params = ("b")
    # test_db_helper.delete(sql, *params)
