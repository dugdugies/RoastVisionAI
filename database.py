import mysql.connector
from mysql.connector import Error

DB = None

def connect():
    global DB
    if DB is None:
        DB = mysql.connector.connect(
            host="localhost",
            user="root",
            password="", 
            database="roasting"
        )
    return DB

def init_db():
    db = connect()
    cursor = db.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INT AUTO_INCREMENT PRIMARY KEY,
            filename VARCHAR(255),
            prediction VARCHAR(50),
            confidence INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    db.commit()
    
def insert_result(filename, prediction, confidence):
    db = connect()
    cursor = db.cursor()
    
    sql = "INSERT INTO results (filename, prediction, confidence) VALUES (%s, %s, %s)"
    cursor.execute(sql, (filename, prediction, confidence))
    
    db.commit()
    
def get_today_results():
    db = connect()
    cursor = db.cursor()
    
    cursor.execute("""
        SELECT filename, prediction, confidence, created_at 
        FROM results 
        WHERE DATE(created_at) = CURDATE()
        ORDER BY id DESC
    """)
    
    return cursor.fetchall()

def get_counts_today():
    db = connect()
    cursor = db.cursor()
    
    cursor.execute("""
        SELECT prediction, COUNT(*) 
        FROM results 
        WHERE DATE(created_at) = CURDATE()
        GROUP BY prediction
    """)
    
    rows = cursor.fetchall()
    
    return {pred: count for pred, count in rows}