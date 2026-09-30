import sqlite3
import threading
from datetime import datetime
import shutil
import os

DB_LOCK = threading.Lock()
DB_PATH = "attendance.db"

def init_db():
    with DB_LOCK, sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY(employee_id) REFERENCES employees(id)
            )
        ''')
        conn.commit()

def add_employee(name):
    with DB_LOCK, sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT OR IGNORE INTO employees (name) VALUES (?)', (name,))
            conn.commit()
        except sqlite3.Error as e:
            print(f"[DB ERROR] Failed to add employee {name}: {e}")

def get_employee_id(name):
    with DB_LOCK, sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM employees WHERE name = ?', (name,))
        row = cursor.fetchone()
        return row[0] if row else None

def log_attendance(name):
    emp_id = get_employee_id(name)
    if emp_id is None:
        add_employee(name)
        emp_id = get_employee_id(name)
    with DB_LOCK, sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('INSERT INTO attendance (employee_id, timestamp) VALUES (?, ?)', (emp_id, timestamp))
        conn.commit()

def get_attendance_records():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT a.id, e.name, a.timestamp
            FROM attendance a
            JOIN employees e ON a.employee_id = e.id
            ORDER BY a.timestamp DESC
        ''')
        return cursor.fetchall()

def get_employees():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM employees ORDER BY name")
        return cursor.fetchall()

def delete_employee(name):
    with DB_LOCK, sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # Получаем ID сотрудника
        cursor.execute("SELECT id FROM employees WHERE name=?", (name,))
        row = cursor.fetchone()
        if row:
            emp_id = row[0]
            # Удаляем посещения
            cursor.execute("DELETE FROM attendance WHERE employee_id=?", (emp_id,))
            # Удаляем сотрудника
            cursor.execute("DELETE FROM employees WHERE id=?", (emp_id,))
        conn.commit()

    # Удаляем папку с фотографиями
    face_dir = os.path.join("face_database", name)
    if os.path.exists(face_dir):
        shutil.rmtree(face_dir)
