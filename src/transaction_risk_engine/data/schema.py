TRANSACTION_ID = "TransactionID"
TARGET = "isFraud"
TIME_COLUMN = "TransactionDT"

REQUIRED_TRANSACTION_COLUMNS = [TRANSACTION_ID, TARGET, TIME_COLUMN]

PROXY_ID_COMPONENTS = {
    "card_uid": ["card1", "card2", "card3", "card5", "card6"],
    "address_uid": ["addr1", "addr2"],
    "email_uid": ["P_emaildomain"],
    "receiver_email_uid": ["R_emaildomain"],
    "device_uid": ["DeviceType", "DeviceInfo", "id_30", "id_31"],
}

PROXY_ID_COLUMNS = list(PROXY_ID_COMPONENTS.keys())

IMPORTANT_COLUMNS_FOR_PROFILING = [
    TRANSACTION_ID, TARGET, TIME_COLUMN, "TransactionAmt", "ProductCD"
] + PROXY_ID_COLUMNS
