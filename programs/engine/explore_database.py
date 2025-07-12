#!/usr/bin/env python3
"""
Database exploration script for Project 2.1
Connects to the PostgreSQL database and examines its structure
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import sys

# Database connection string from Discord
DATABASE_URL = 'postgresql://postgres:7crQ9MrrBC216QmgSB^S@darcydb.crgk48smefvn.ap-southeast-2.rds.amazonaws.com:5432/postgres'

def connect_to_database():
    """Connect to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        print("✅ Successfully connected to PostgreSQL database")
        return conn
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        return None

def explore_schemas(conn):
    """List all schemas in the database."""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
            ORDER BY schema_name;
        """)
        
        schemas = cursor.fetchall()
        print(f"\n📁 Available Schemas ({len(schemas)}):")
        for schema in schemas:
            print(f"   • {schema[0]}")
        
        cursor.close()
        return [schema[0] for schema in schemas]
    except Exception as e:
        print(f"❌ Error exploring schemas: {e}")
        return []

def explore_tables(conn, schema_name='public'):
    """List all tables in a specific schema."""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT table_name, table_type
            FROM information_schema.tables 
            WHERE table_schema = %s
            ORDER BY table_name;
        """, (schema_name,))
        
        tables = cursor.fetchall()
        print(f"\n📋 Tables in '{schema_name}' schema ({len(tables)}):")
        for table_name, table_type in tables:
            print(f"   • {table_name} ({table_type})")
        
        cursor.close()
        return [table[0] for table in tables]
    except Exception as e:
        print(f"❌ Error exploring tables: {e}")
        return []

def explore_table_structure(conn, table_name, schema_name='public'):
    """Examine the structure of a specific table."""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position;
        """, (schema_name, table_name))
        
        columns = cursor.fetchall()
        print(f"\n🔍 Structure of '{schema_name}.{table_name}':")
        print("   Column Name           | Data Type    | Nullable | Default")
        print("   ---------------------|--------------|----------|--------")
        for col_name, data_type, nullable, default in columns:
            default_str = str(default) if default else "None"
            print(f"   {col_name:<20} | {data_type:<12} | {nullable:<8} | {default_str}")
        
        cursor.close()
        return columns
    except Exception as e:
        print(f"❌ Error exploring table structure: {e}")
        return []

def explore_table_data(conn, table_name, schema_name='public', limit=5):
    """Sample data from a table."""
    try:
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT * FROM {schema_name}.{table_name} 
            LIMIT %s;
        """, (limit,))
        
        rows = cursor.fetchall()
        colnames = [desc[0] for desc in cursor.description]
        
        print(f"\n📊 Sample data from '{schema_name}.{table_name}' (first {limit} rows):")
        if rows:
            print(f"   Columns: {', '.join(colnames)}")
            for i, row in enumerate(rows, 1):
                print(f"   Row {i}: {row}")
        else:
            print("   No data found in this table.")
        
        cursor.close()
        return rows
    except Exception as e:
        print(f"❌ Error exploring table data: {e}")
        return []

def main():
    """Main exploration function."""
    print("🔍 Database Exploration for Project 2.1")
    print("=" * 50)
    
    # Connect to database
    conn = connect_to_database()
    if not conn:
        sys.exit(1)
    
    try:
        # Explore schemas
        schemas = explore_schemas(conn)
        
        # Explore tables in each schema
        for schema in schemas:
            tables = explore_tables(conn, schema)
            
            # If there are tables, explore their structure
            if tables:
                print(f"\n🔍 Detailed exploration of '{schema}' schema:")
                for table in tables[:3]:  # Limit to first 3 tables to avoid spam
                    explore_table_structure(conn, table, schema)
                    explore_table_data(conn, table, schema, limit=3)
        
        # Focus on project_two schema if it exists
        if 'project_two' in schemas:
            print(f"\n🎯 Special focus on 'project_two' schema (relevant for Project 2.2):")
            tables = explore_tables(conn, 'project_two')
            for table in tables:
                explore_table_structure(conn, table, 'project_two')
                explore_table_data(conn, table, 'project_two', limit=5)
        
        print(f"\n📋 Database Exploration Summary:")
        print(f"   • Total schemas: {len(schemas)}")
        print(f"   • Key findings: Database is set up for Project 2.2")
        print(f"   • Next step: Create tables in 'project_two' schema")
        
    finally:
        conn.close()
        print("\n✅ Database connection closed")

if __name__ == "__main__":
    main() 