import sqlite3

def get_item_name(typeid):
    try:
        # Connect to the database
        conn = sqlite3.connect('osrsmarketdata.sqlite')
        # Create a cursor
        cur = conn.cursor()
        # Query the mapping table for the name corresponding to the typeid
        cur.execute("SELECT name, buylimit FROM Mapping WHERE typeid = ?", (typeid,))
        # Fetch the result
        result = cur.fetchone()
        # Close the connection
        conn.close()
        if result:
            print(result[1])
            return result[0]
        else:
            return None
    except sqlite3.Error as e:
        print("Database error:", e)
        return None

def get_item_id(name):
    try:
        # Connect to the database
        conn = sqlite3.connect('osrsmarketdata.sqlite')
        # Create a cursor
        cur = conn.cursor()
        # Query the mapping table for the typeid corresponding to the name
        # Using TRIM to ignore leading and trailing spaces and COLLATE NOCASE for case-insensitive comparison
        cur.execute("SELECT typeid FROM Mapping WHERE TRIM(name) = TRIM(?) COLLATE NOCASE", (name,))
        # Fetch the result
        result = cur.fetchone()
        # Close the connection
        conn.close()
        if result:
            return result[0]
        else:
            return None
    except sqlite3.Error as e:
        print("Database error:", e)
        return None

# Get item name from user input
name_input = "Soulreaper axe".strip()
if not name_input:
    print("Please enter a valid item name.")
else:
    item_id = get_item_id(name_input)
    if item_id:
        print("Item ID:", item_id)
    else:
        print("No item found with name", name_input)
        
# Get typeid from user input
typeid_input = 29993
try:
    typeid = int(typeid_input)
except ValueError:
    print("Please enter a valid integer for the item ID.")
else:
    name = get_item_name(typeid)
    if name:
        print("Item name:", name)
    else:
        print("No item found with typeid", typeid)