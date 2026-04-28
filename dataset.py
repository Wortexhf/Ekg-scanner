import wfdb

def ecg_record(record_id):
    record_id = str(record_id).strip()
    if not record_id:
        print("Error: Enter the correct record ID.")
        return None, None

    try:
        header = wfdb.rdheader(record_id, pn_dir='mitdb')
        sampto = header.fs * 60 
        record = wfdb.rdrecord(record_id, pn_dir='mitdb', sampto=sampto)
        ann = wfdb.rdann(record_id, 'atr', pn_dir='mitdb', sampto=sampto)
        return record, ann
    except ValueError as e:
        print(f"Record '{record_id}' not found. Error: {e}")
        return None, None
    except Exception as e:
        print(f"Unexpected error while loading '{record_id}': {e}")
        return None, None