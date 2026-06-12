import warnings
import os
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv('D2C_DATA_PATH', ROOT_DIR / 'data'))
SNAPSHOT_FILE = DATA_DIR / 'rfm_modeling_snapshot.csv'
LABEL_FILE = DATA_DIR / 'churn_labels.csv'

print('DATA_DIR:', DATA_DIR)
print('SNAPSHOT_FILE exists:', SNAPSHOT_FILE.exists())
print('LABEL_FILE exists:', LABEL_FILE.exists())

snapshot_df = pd.read_csv(SNAPSHOT_FILE)
labels_df = pd.read_csv(LABEL_FILE)

# merge and validate
merged_df = snapshot_df.merge(
    labels_df[['customer_id', 'churn_next_60d', 'split']].drop_duplicates('customer_id'),
    on='customer_id',
    how='left',
    validate='one_to_one',
)
missing_labels = merged_df['churn_next_60d'].isna().sum()
if missing_labels > 0:
    print(f'MISSING_LABELS: {missing_labels}')
else:
    print('No missing labels.')

print('Merged shape:', merged_df.shape)
print('Train/validation/test distribution:')
print(merged_df['split'].value_counts(dropna=False))
print('Churn rate overall:', merged_df['churn_next_60d'].mean())

# snapshot date
snapshot_date = None
try:
    snap_dates = pd.to_datetime(snapshot_df['snapshot_date'].unique())
    if len(snap_dates) != 1:
        warnings.warn(f"Multiple snapshot_date values found: {snap_dates}")
    snapshot_date = snap_dates[0]
    print('Snapshot reference date:', snapshot_date.date())
except Exception as e:
    print('Could not infer snapshot_date:', e)

# label snapshot_date
try:
    label_snap_dates = pd.to_datetime(labels_df['snapshot_date'].unique())
    if len(label_snap_dates) != 1:
        warnings.warn(f"Multiple label snapshot_date values found: {label_snap_dates}")
    else:
        if snapshot_date is not None and label_snap_dates[0] != snapshot_date:
            print('SNAPSHOT MISMATCH: snapshot vs labels', snapshot_date.date(), label_snap_dates[0].date())
        else:
            print('Label snapshot date OK')
except Exception as e:
    print('Label snapshot date check failed:', e)

# orders post-snapshot
orders_path = DATA_DIR / 'orders.csv'
if orders_path.exists():
    orders = pd.read_csv(orders_path, parse_dates=['order_date'])
    if snapshot_date is not None:
        post_snapshot = orders[orders['order_date'] > snapshot_date]
        print('Orders rows after snapshot:', len(post_snapshot))
        if len(post_snapshot) > 0:
            print('WARNING: orders.csv contains post-snapshot rows. Do NOT use post-snapshot orders as features.')
else:
    print('orders.csv not found; skipping orders post-snapshot check')

# support tickets
support_path = DATA_DIR / 'support_tickets.csv'
if support_path.exists():
    tickets = pd.read_csv(support_path, parse_dates=['ticket_date'])
    if snapshot_date is not None:
        bad = tickets[tickets['ticket_date'] > snapshot_date]
        print('Support tickets after snapshot:', len(bad))
        if len(bad) > 0:
            print('WARNING: support_tickets.csv contains rows after the snapshot date')
else:
    print('support_tickets.csv not found; skipping')

# web events
web_path = DATA_DIR / 'web_events_snapshot.csv'
if web_path.exists():
    web = pd.read_csv(web_path, parse_dates=['snapshot_date'])
    unique_web_dates = web['snapshot_date'].unique()
    print('web_events_snapshot snapshot_date unique values:', unique_web_dates)
    if snapshot_date is not None and any(pd.to_datetime(unique_web_dates) != snapshot_date):
        print('WARNING: web_events_snapshot contains snapshot_date values different from the main snapshot')
else:
    print('web_events_snapshot.csv not found; skipping')

# suspicious columns
suspicious = [c for c in snapshot_df.columns if any(k in c.lower() for k in ['future', 'post', 'next_', 'after_snapshot'])]
print('Suspicious column names indicating potential leakage:', suspicious)

print('\nExtended leakage checks complete.')
