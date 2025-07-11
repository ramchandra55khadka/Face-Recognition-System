
import sqlite3
import pickle
import os
from typing import List, Tuple, Optional
from datetime import datetime
import logging
import numpy as np

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Enhanced database manager with additional features and schema migration."""
    
    def __init__(self, db_path: str = "faces.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.migrate_schema()
        self.create_tables()
    
    def check_column_exists(self, table: str, column: str) -> bool:
        """Check if a column exists in a table."""
        try:
            cursor = self.conn.execute(f"PRAGMA table_info({table})")
            columns = [info[1] for info in cursor.fetchall()]
            return column in columns
        except Exception as e:
            logger.error(f"Error checking column {column} in {table}: {e}")
            return False
    
    def migrate_schema(self):
        """Migrate the database schema to include new columns if they don't exist."""
        try:
            # Check if users table exists
            cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
            if not cursor.fetchone():
                return  # Table doesn't exist, will be created in create_tables
            
            # Add missing columns if they don't exist
            if not self.check_column_exists('users', 'dob'):
                self.conn.execute("ALTER TABLE users ADD COLUMN dob TEXT")
                logger.info("Added dob column to users table")
            
            if not self.check_column_exists('users', 'gender'):
                self.conn.execute("ALTER TABLE users ADD COLUMN gender TEXT")
                logger.info("Added gender column to users table")
            
            if not self.check_column_exists('users', 'voter_id'):
                self.conn.execute("ALTER TABLE users ADD COLUMN voter_id TEXT")
                logger.info("Added voter_id column to users table")
            
            # Ensure voter_id has UNIQUE constraint
            # SQLite doesn't support modifying constraints directly, so we need to recreate the table
            cursor = self.conn.execute("PRAGMA table_info(users)")
            columns = [info[1] for info in cursor.fetchall()]
            if 'voter_id' in columns:
                try:
                    # Check if voter_id has UNIQUE constraint
                    cursor = self.conn.execute("PRAGMA index_list(users)")
                    indexes = [info[1] for info in cursor.fetchall()]
                    if 'sqlite_autoindex_users_1' not in indexes:  # Check for UNIQUE constraint
                        # Create temporary table
                        self.conn.execute("""
                            CREATE TABLE users_temp (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                name TEXT NOT NULL,
                                face_embedding BLOB NOT NULL,
                                image_path TEXT NOT NULL,
                                thumbnail_path TEXT,
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                is_active BOOLEAN DEFAULT 1,
                                dob TEXT,
                                gender TEXT,
                                voter_id TEXT UNIQUE
                            )
                        """)
                        # Copy data
                        self.conn.execute("""
                            INSERT INTO users_temp 
                            SELECT id, name, face_embedding, image_path, thumbnail_path, 
                                   created_at, updated_at, is_active, dob, gender, voter_id 
                            FROM users
                        """)
                        # Drop old table and rename new one
                        self.conn.execute("DROP TABLE users")
                        self.conn.execute("ALTER TABLE users_temp RENAME TO users")
                        logger.info("Added UNIQUE constraint to voter_id")
                    self.conn.commit()
                except Exception as e:
                    logger.error(f"Error applying UNIQUE constraint to voter_id: {e}")
                    self.conn.rollback()
                    raise
        except Exception as e:
            logger.error(f"Schema migration error: {e}")
            self.conn.rollback()
            raise
    
    def create_tables(self):
        """Create database tables."""
        try:
            # Users table with new fields
            users_query = """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                face_embedding BLOB NOT NULL,
                image_path TEXT NOT NULL,
                thumbnail_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                dob TEXT,
                gender TEXT,
                voter_id TEXT UNIQUE
            )
            """
            
            # Recognition logs table
            logs_query = """
            CREATE TABLE IF NOT EXISTS recognition_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                similarity_score REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """
            
            # Create indexes for better performance
            index_query = """
            CREATE INDEX IF NOT EXISTS idx_users_name ON users(name);
            CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active);
            CREATE INDEX IF NOT EXISTS idx_users_voter_id ON users(voter_id) WHERE voter_id IS NOT NULL;
            CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON recognition_logs(timestamp);
            """
            
            self.conn.execute(users_query)
            self.conn.execute(logs_query)
            self.conn.executescript(index_query)
            self.conn.commit()
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Database table creation error: {e}")
            raise
    
    def add_user(self, name: str, embedding: np.ndarray, image_path: str, 
                 thumbnail_path: Optional[str] = None, dob: Optional[str] = None, 
                 gender: Optional[str] = None, voter_id: Optional[str] = None) -> int:
        """Add a new user to the database."""
        try:
            emb_blob = pickle.dumps(embedding)
            query = """
            INSERT INTO users (name, face_embedding, image_path, thumbnail_path, dob, gender, voter_id) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            cursor = self.conn.execute(query, (name, emb_blob, image_path, thumbnail_path, dob, gender, voter_id))
            self.conn.commit()
            user_id = cursor.lastrowid
            logger.info(f"User '{name}' added with ID {user_id}")
            return user_id
        except Exception as e:
            logger.error(f"Error adding user: {e}")
            raise
    
    def get_all_users(self, active_only: bool = True) -> List[Tuple]:
        """Get all users from the database."""
        try:
            query = """
            SELECT id, name, face_embedding, image_path, thumbnail_path, created_at, dob, gender, voter_id 
            FROM users 
            WHERE is_active = ? OR ? = 0
            ORDER BY created_at DESC
            """
            cursor = self.conn.execute(query, (1 if active_only else 0, 1 if active_only else 0))
            results = cursor.fetchall()
            
            users = []
            for row in results:
                try:
                    embedding = pickle.loads(row[2])
                    users.append((row[0], row[1], embedding, row[3], row[4], row[5], row[6], row[7], row[8]))
                except Exception as e:
                    logger.warning(f"Failed to load embedding for user {row[0]}: {e}")
            
            return users
        except Exception as e:
            logger.error(f"Error getting users: {e}")
            raise
    
    def get_user_by_id(self, user_id: int) -> Optional[Tuple]:
        """Get a specific user by ID."""
        try:
            query = """
            SELECT id, name, face_embedding, image_path, thumbnail_path, created_at, dob, gender, voter_id 
            FROM users 
            WHERE id = ? AND is_active = 1
            """
            cursor = self.conn.execute(query, (user_id,))
            result = cursor.fetchone()
            
            if result:
                embedding = pickle.loads(result[2])
                return (result[0], result[1], embedding, result[3], result[4], result[5], result[6], result[7], result[8])
            return None
        except Exception as e:
            logger.error(f"Error getting user by ID: {e}")
            return None
    
    def search_users(self, query: str) -> List[Tuple]:
        """Search users by name or voter ID."""
        try:
            search_query = """
            SELECT id, name, face_embedding, image_path, thumbnail_path, created_at, dob, gender, voter_id 
            FROM users 
            WHERE (name LIKE ? OR voter_id LIKE ?) AND is_active = 1
            ORDER BY name
            """
            cursor = self.conn.execute(search_query, (f'%{query}%', f'%{query}%'))
            results = cursor.fetchall()
            
            users = []
            for row in results:
                try:
                    embedding = pickle.loads(row[2])
                    users.append((row[0], row[1], embedding, row[3], row[4], row[5], row[6], row[7], row[8]))
                except Exception as e:
                    logger.warning(f"Failed to load embedding for user {row[0]}: {e}")
            
            return users
        except Exception as e:
            logger.error(f"Error searching users: {e}")
            return []
    
    def deactivate_user(self, user_id: int) -> bool:
        """Deactivate a user (soft delete)."""
        try:
            query = "UPDATE users SET is_active = 0, updated_at = CURRENT_TIMESTAMP WHERE id = ?"
            cursor = self.conn.execute(query, (user_id,))
            self.conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error deactivating user: {e}")
            return False
    
    def log_recognition(self, user_id: int, similarity_score: float):
        """Log a recognition event."""
        try:
            query = "INSERT INTO recognition_logs (user_id, similarity_score) VALUES (?, ?)"
            self.conn.execute(query, (user_id, similarity_score))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error logging recognition: {e}")
    
    def get_recognition_stats(self, days: int = 30) -> List[Tuple]:
        """Get recognition statistics for the last N days."""
        try:
            query = """
            SELECT u.name, u.voter_id, COUNT(r.id) as recognition_count, AVG(r.similarity_score) as avg_similarity
            FROM users u
            LEFT JOIN recognition_logs r ON u.id = r.user_id 
            WHERE r.timestamp > datetime('now', '-{} days')
            GROUP BY u.id, u.name, u.voter_id
            ORDER BY recognition_count DESC
            """.format(days)
            
            cursor = self.conn.execute(query)
            return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error getting recognition stats: {e}")
            return []
    
    def get_user_count(self) -> int:
        """Get the total number of active users."""
        try:
            cursor = self.conn.execute("SELECT COUNT(*) FROM users WHERE is_active = 1")
            return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Error getting user count: {e}")
            return 0
    
    def close(self):
        """Close the database connection."""
        self.conn.close()