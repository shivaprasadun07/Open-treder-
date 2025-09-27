import sqlite3
import os
import time

class StateManager:
    def __init__(self, db_path='data/trades.db'):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                close_price REAL,
                amount REAL NOT NULL,
                leverage INTEGER NOT NULL,
                atr REAL NOT NULL,
                sl_price REAL,
                status TEXT NOT NULL, -- 'OPEN' or 'CLOSED'
                open_timestamp INTEGER NOT NULL,
                close_timestamp INTEGER
            )
        ''')
        self.conn.commit()

    def add_trade(self, pair, side, entry_price, amount, leverage, atr):
        cursor = self.conn.cursor()
        timestamp = int(time.time())
        cursor.execute('''
            INSERT INTO trades (pair, side, entry_price, amount, leverage, atr, status, open_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (pair, side, entry_price, amount, leverage, atr, 'OPEN', timestamp))
        self.conn.commit()
        return cursor.lastrowid

    def close_trade(self, trade_id, close_price):
        cursor = self.conn.cursor()
        timestamp = int(time.time())
        cursor.execute('''
            UPDATE trades
            SET status = 'CLOSED', close_price = ?, close_timestamp = ?
            WHERE id = ?
        ''', (close_price, timestamp, trade_id))
        self.conn.commit()

    def update_sl_price(self, trade_id, sl_price):
        cursor = self.conn.cursor()
        cursor.execute('UPDATE trades SET sl_price = ? WHERE id = ?', (sl_price, trade_id))
        self.conn.commit()

    def get_open_trades(self):
        cursor = self.conn.cursor()
        cursor.row_factory = sqlite3.Row # Allows accessing columns by name
        cursor.execute("SELECT * FROM trades WHERE status = 'OPEN'")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]