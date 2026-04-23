import wfdb 

def ecg_record(record_id):
    record_id = str(record_id).strip()
    if not record_id:
        print("Error. Enter the correct id")
        return None, None 
    try:
        record = wfdb.rdrecord(record_id, pn_dir='mitdb') 
        ann = wfdb.rdann(record_id, 'atr', pn_dir='mitdb')
        return record, ann
    except ValueError as e:
        print(f'{record_id} not found. Error:{e}')
        return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None