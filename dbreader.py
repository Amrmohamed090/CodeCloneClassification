import psycopg2
from psycopg2 import Error

def print_clones():
    try:
        conn = psycopg2.connect(
            dbname="bigclonebench",
            user="postgres",
            password="123",
            host="localhost"
        )
        
        cursor = conn.cursor()
        
        query = """
        SELECT 
            c.function_id_one,
            c.function_id_two,
            c.type,
            c.similarity_token,
            f1.text as code_one,
            f2.text as code_two
        FROM clones c
        JOIN pretty_printed_functions f1 ON c.function_id_one = f1.function_id
        JOIN pretty_printed_functions f2 ON c.function_id_two = f2.function_id
        WHERE c.min_confidence >= 0.7
        AND c.min_judges >= 2
        AND NOT EXISTS (
            SELECT 1 FROM false_positives fp 
            WHERE fp.function_id_one = c.function_id_one 
            AND fp.function_id_two = c.function_id_two
        )
        LIMIT 5;
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        for row in rows:
            print(f"\nClone Pair (Type-{row[2]}, Similarity: {row[3]:.2f})")
            print("\nFunction 1 (ID: {}):\n{}".format(row[0], row[4]))
            print("\nFunction 2 (ID: {}):\n{}".format(row[1], row[5]))
            print("-" * 80)
            
    except (Exception, Error) as error:
        print("Error:", error)
    finally:
        if conn:
            cursor.close()
            conn.close()

if __name__ == "__main__":
    print_clones()