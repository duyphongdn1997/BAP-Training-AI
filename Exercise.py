from mysql import connector

class DBHelper():
    

    def __init__(self, host, port, user, passwd, db):
        self.__host = host
        self.__port = port
        self.__user = user
        self.__passwd = passwd
        self.__db = db

    def connectMysql(self):
        conn=connector.connect(host=self.__host,
                            port=self.__port,
                            user=self.__user,
                            passwd=self.__passwd,
                            #db=self.__db, #Do not specify the database name
                            charset='utf8') 
        return conn

    def connectDatabase(self):
        conn = connector.connect(host=self.__host,
                                    user=self.__user,
                                    password=self.__passwd,
                                    port=self.__port,
                                    database=self.__db,
                                    charset='utf8')
        return conn

    def createDatabase(self):
        conn = self.connectMysql()
        sql = 'create database if not exists ' + self.db
        cur = conn.cursor()
        cur.execute(sql)
        cur.close()
        conn.close()

    def createTable(self, sql):
        conn = self.connectDatabase()
        cur = conn.cursor()
        cur.execute(sql)
        cur.close()
        conn.close()

    def insert(self,sql,*params):
        conn = self.connectDatabase()
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
        cur.close()
        conn.close()

    def update(self,sql,*params):
        conn = self.connectDatabase()
        cur = conn.cursor()
        cur.execute(sql,params)
        conn.commit()
        cur.close()
        conn.close()

    def close(self, sql, *param):
        conn = self.connectDatabase()
        conn.close()

    def delete(self,sql,*params):
        conn = self.connectDatabase()
        cur = conn.cursor()
        cur.execute(sql,params)
        conn.commit()
        cur.close()
        conn.close()

class Person:
    def __init__(self, name: str, age: int, address: str):
        self.__name = name
        self.__age = age
        self.__address = address

    def getName(self): 
        return self.__name
    def getAge(self): 
        return self.__age
    def getAddress(self): 
        return self.__address

if __name__ == '__main__': 

    person0 = Person(name='a', age=1, address='10 abc')
    person1 = Person(name='b', age=2, address='11 def')
    person2 = Person(name='c', age=3, address='01 ghi')
    
    # connect SQL
    test_db_helper = DBHelper('192.168.1.254','3306','root','Nhatquan1', 'Test')
    test_db_helper.connectMysql() # connect SQL
    test_db_helper.createDatabase() # create Database Test
    test_db_helper.connectDatabase() # connect database Test

    #create Table Person
    sql = 'create table Person(name nvarchar(50) primary key, age integer, NgaySinh nvarchar(20))'
    test_db_helper.createTable(sql)

    #insert data
    sql = 'insert into Person(name, age, Address) values(%s,%s,%s,%s)'
    test_db_helper.insert(sql, (person0.getName(), person0.getAge(), person0.getAddress()))

    #update data
    sql = 'update Person set age = %s, address = %s where name = %s'
    params=(person1.getAge(), person1.getAddress(), person1.getName())
    test_db_helper.update(sql,*params)

    #delete data
    sql = 'delete from Person where name = %s'
    params = ("b")
    test_db_helper.delete(sql, *params)