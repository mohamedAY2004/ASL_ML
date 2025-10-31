import sqlite3
from datetime import datetime

class DatabaseHandler:
    def __init__(self, db_name="asl_letters.db"):
        self.db_name = db_name
        self.create_table()
    
    def create_table(self):
        """Create the letters table if it doesn't exist"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS saved_letters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                letter TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_letter(self, letter):
        """Save a letter to the database"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO saved_letters (letter, timestamp)
            VALUES (?, ?)
        ''', (letter, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        conn.commit()
        conn.close()
    
    def get_saved_letters(self):
        """Retrieve all saved letters"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM saved_letters ORDER BY timestamp DESC')
        letters = cursor.fetchall()
        
        conn.close()
        return letters 