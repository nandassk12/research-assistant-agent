"""
utils/memory.py

SQLite-based persistent storage for the Personal Research Assistant Agent.
Stores and retrieves search results (queries and JSON) locally.
"""

import sqlite3
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

DB_FILE = "research_memory.db"

logger = logging.getLogger(__name__)


def init_db() -> None:
    """
    Initialises the SQLite database and creates the searches table if it 
    does not exist. Fails silently on exceptions.
    """
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS searches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL UNIQUE,
                    result_json TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
    except Exception as exc:
        logger.warning(f"Failed to initialize database: {exc}")


def save_search(query: str, result: Dict[str, Any]) -> None:
    """
    Saves a search query and its JSON result to the database.
    Keeps only the last 20 searches, deleting the oldest if exceeding.
    """
    try:
        init_db()
        result_str = json.dumps(result)
        
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            
            # Upsert the query (update if exists, insert if new)
            cursor.execute('''
                INSERT INTO searches (query, result_json, timestamp)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(query) DO UPDATE SET
                    result_json=excluded.result_json,
                    timestamp=CURRENT_TIMESTAMP
            ''', (query, result_str))
            
            # Prune to keep only the 20 most recent
            cursor.execute('''
                DELETE FROM searches
                WHERE id NOT IN (
                    SELECT id FROM searches
                    ORDER BY timestamp DESC
                    LIMIT 20
                )
            ''')
            conn.commit()
            logger.info(f"Saved search for query: {query[:30]}...")
            
    except Exception as exc:
        logger.warning(f"Failed to save search: {exc}")


def get_search(query: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a cached search result from the database by query string.
    Returns the deserialised dictionary, or None if not found/error.
    """
    try:
        init_db()
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT result_json 
                FROM searches 
                WHERE query = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            ''', (query,))
            row = cursor.fetchone()
            
            if row:
                return json.loads(row[0])
    except Exception as exc:
        logger.warning(f"Failed to retrieve search: {exc}")
        
    return None


def load_searches(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Loads recent searches from the database, up to the specified limit.
    Returns a list of dictionaries with query, result, and timestamp.
    """
    searches = []
    try:
        init_db()
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT query, result_json, timestamp
                FROM searches
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            rows = cursor.fetchall()
            
            for row in rows:
                searches.append({
                    "query": row[0],
                    "result": json.loads(row[1]),
                    "timestamp": row[2]
                })
    except Exception as exc:
        logger.warning(f"Failed to load searches: {exc}")
        
    return searches


def get_recent_queries(limit: int = 5) -> List[str]:
    """
    Returns a list of recent query strings, deduplicated and ordered by time.
    """
    try:
        recent = load_searches(limit=limit * 2)
        
        # Deduplicate while preserving order
        queries = []
        seen = set()
        for item in recent:
            q = item["query"]
            if q not in seen:
                seen.add(q)
                queries.append(q)
                if len(queries) >= limit:
                    break
                    
        return queries
    except Exception as exc:
        logger.warning(f"Failed to get recent queries: {exc}")
        return []


def delete_search(query: str) -> None:
    """
    Deletes a specific search by query from the database.
    """
    try:
        init_db()
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM searches WHERE query = ?', (query,))
            conn.commit()
    except Exception as exc:
        logger.warning(f"Failed to delete search: {exc}")


def clear_history() -> None:
    """
    Clears all rows from the searches table.
    """
    try:
        init_db()
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM searches')
            conn.commit()
            logger.info("Cleared all search history from database")
    except Exception as exc:
        logger.warning(f"Failed to clear history: {exc}")


def get_db_stats() -> Dict[str, Any]:
    """
    Returns statistics about the SQLite database (count, timestamps, size).
    """
    stats = {}
    try:
        init_db()
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM searches')
            count = cursor.fetchone()[0]
            
            cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM searches')
            min_ts, max_ts = cursor.fetchone()
            
            size_kb = 0.0
            if os.path.exists(DB_FILE):
                size_kb = os.path.getsize(DB_FILE) / 1024.0
                
            stats = {
                "total_searches": count,
                "oldest": min_ts,
                "newest": max_ts,
                "db_size_kb": size_kb
            }
    except Exception as exc:
        logger.warning(f"Failed to get db stats: {exc}")
        
    return stats
