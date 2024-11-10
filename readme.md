# BigCloneBench Setup Guide

## Prerequisites
- PostgreSQL installed and running
- Python with required dependencies
- Git

## Database Setup Instructions

### 1. Download BigCloneBench
```bash
git clone https://github.com/clonebench/BigCloneBench.git
```

### 2. Extract Database File
- Locate the `bcb` file in the downloaded repository
- If compressed, extract it to your working directory

### 3. Database Creation
Navigate to the directory containing the `bcb` file and run the following commands:

```bash
# Create new database
createdb -U postgres bigclonebench

# Import the database schema and data
psql -U postgres -d bigclonebench -f bcb

# Verify tables were created correctly
psql -U postgres -d bigclonebench \dt
```

**Note:** Replace `postgres` with your PostgreSQL username if different.

## SiamesCC Configuration

### Update Database Connection
In `SiamesCC/training.py`, locate the main function and update the database connection string with your credentials:

```python
engine = create_engine("postgresql+psycopg2://postgres:{your_password}@localhost/bigclonebench")
```

Replace `{your_password}` with your actual PostgreSQL password.

## Troubleshooting

If you encounter authentication issues:
1. Verify your PostgreSQL username and password
2. Ensure PostgreSQL service is running
3. Check that you have proper permissions to create and modify databases

## Additional Resources
- [BigCloneBench Repository](https://github.com/clonebench/BigCloneBench)
- PostgreSQL Documentation
