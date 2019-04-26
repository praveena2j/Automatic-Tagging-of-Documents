#!/usr/bin/python
import re, MySQLdb
class DBUtils:
    def __init__(self, aHost='impdevenv.cloudapp.net', aUser='root', aPassword='0$c!ent', aPort=3306, aDB='oscient'):
        self.connection = None
        self.cursor     = None
        self.host       = aHost
        self.user       = aUser
        self.password   = aPassword
        self.port       = aPort
        self.db         = aDB

    def makeDBConnection(self):
        # Open the database connection and prepare the cursor
        self.connection = MySQLdb.connect(host=self.host, user=self.user, passwd=self.password, port=self.port, db=self.db)
        self.connection.ping(True)
        self.cursor = self.connection.cursor()

    def closeDBConnection(self):
        # Close the cursor object and connection
        self.cursor.close()
        self.connection.close()

    def handleQuery(self, query, *params):
        try:
            self.cursor = self.connection.cursor()
            if len(params) > 0:
                self.cursor.execute(query % params)
            else:
                self.cursor.execute(query)
        except (AttributeError, MySQLdb.OperationalError):
            print "*** MySQL Connection Exception Handler ***"
            self.makeDBConnection()
            if len(params) > 0:
                self.cursor.execute(query % params)
            else:
                self.cursor.execute(query)
        return self.cursor

    def handleQuery1():
        makeDBConnection("localhost", "root", "password", 3336, "oscient")
        query = "SELECT * from topic_validations where query = '%s'"
        gCursor.execute(query % gQuery)
        rows = gCursor.fetchone()
        tags = ''
        if rows:
            print html_error
            tags = rows[2]
        else:
            tags = getTags(gQuery)
            query = "INSERT INTO topic_validations (`query`, `tags`, `newtags`,`changed`) values ('%s', '%s', '',0)"
            gCursor.execute(query % (gQuery, tags))
            gConnection.commit()
        print html_question % gQuery
        print re.sub(',', '<BR>', tags)
        closeDBConnection()

