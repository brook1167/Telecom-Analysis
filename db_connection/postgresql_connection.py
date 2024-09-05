import pandas as pd
from sqlalchemy import create_engine

def fetch_data_from_postgres():
   
    # Database credentials
    dbname = "telecom_db"
    user = "postgres"
    password = "12345"
    host = "localhost"
    port = "5432"
    
    # SQL query
    sql_query = "SELECT * FROM xdr_data"
    
    # Construct the SQLAlchemy connection string
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    
    try:
        # Create a SQLAlchemy engine
        engine = create_engine(connection_string)
        
        # Use Pandas to read SQL query results into a DataFrame
        df = pd.read_sql(sql_query, engine)
        
        # Return the DataFrame
        return df
    
    except Exception as e:
        # Print the error message and return None
        print(f"An error occurred: {e}")
        return None