import sqlite3
import hashlib

# Create a connection to the database
conn = sqlite3.connect('users.db')
c = conn.cursor()

# Create the table if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS users
             (username text, password_hash text)''')

# Insert the default usernames and hashed passwords
users = [('admin', hashlib.sha256(b'admin123').hexdigest()),
         ('security', hashlib.sha256(b'security123').hexdigest())]
c.executemany('INSERT INTO users VALUES (?, ?)', users)

# Commit the changes and close the connection
conn.commit()
conn.close()