import datetime
import json
import os
import os.path as osp
import yaml

import concurrent.futures
import dateutil.parser
import pandas
import tabulate


def summarize_log(logs_dir, name, keys, target_key, objective,
                  show_range=True):
    try:
        with open(osp.join(logs_dir, name, 'params.yaml')) as f:
            params = yaml.safe_load(f)
    except Exception:
        return None, osp.join(logs_dir, name)

    log_file = osp.join(logs_dir, name, 'log')
    try:
        with open(log_file) as f:
            df = pandas.DataFrame(json.load(f))
    except Exception:
        return None, osp.join(logs_dir, name)

    try:
        if objective == 'min':
            idx = df[target_key].idxmin()
        else:
            idx = df[target_key].idxmax()
    except Exception:
        idx = None

    if idx in [None, float('nan')]:
        dfi = None
    else:
        dfi = df.ix[idx]

    row = []
    for key in keys:
        if key == 'name':
            row.append(name)
        elif key in ['epoch', 'iteration']:
            if dfi is None:
                value = None
            else:
                value = '%d' % dfi[key]
            max_value = df[key].max()
            row.append('%s /%d' % (value, max_value))
        elif key.endswith('/loss'):
            if dfi is None:
                value = None
            else:
                value = '%.3f' % dfi[key]
            min_value = df[key].min()
            max_value = df[key].max()
            if show_range:
                row.append('%.3f< %s <%.3f' % (min_value, value, max_value))
            else:
                row.append('%s' % value)
        elif '/' in key:
            if dfi is None or key in dfi:
                min_value = max_value = None
                if dfi is None:
                    value = None
                else:
                    value = '%.3f' % dfi[key]
                if df is not None and key in df:
                    min_value = '%.3f' % df[key].min()
                    max_value = '%.3f' % df[key].max()
                if show_range:
                    row.append('%s< %s <%s' % (min_value, value, max_value))
                else:
                    row.append('%s' % value)
            else:
                row.append(None)
        elif key == 'last_time':
            if df is None:
                value = None
            else:
                elapsed_time = df['elapsed_time'].max()
                value = params.get('timestamp', None)
                if value is not None:
                    value = dateutil.parser.parse(value)
                    value += datetime.timedelta(seconds=elapsed_time)
                    now = datetime.datetime.now()
                    value = now - value
                    value -= datetime.timedelta(
                        microseconds=value.microseconds)
                    value = max(datetime.timedelta(seconds=0), value)
                    value = '- %s' % value.__str__()
            row.append(value)
        elif key in params:
            row.append(params[key])
        elif dfi is not None and key in dfi:
            row.append(dfi[key])
        else:
            row.append(None)
    return row, None


def summarize_logs(logs_dir, keys, target_key, objective, sort=None,
                   show_range=True, as_df=False, key_remap=None):
    assert objective in ['min', 'max']
    assert target_key in keys
    if key_remap is None:
        key_remap = {}

    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        for name in os.listdir(logs_dir):
            future = executor.submit(
                summarize_log,
                logs_dir=logs_dir,
                name=name,
                keys=keys,
                target_key=target_key,
                objective=objective,
                show_range=show_range,
            )
            futures.append(future)

    rows = []
    ignored = []
    for future in futures:
        row, log_dir_ignored = future.result()
        if log_dir_ignored:
            ignored.append(log_dir_ignored)
            continue
        rows.append(row)

    if as_df:
        df = pandas.DataFrame(data=rows, columns=keys)
        return df

    if sort is None:
        sort = [keys[0]]

    for k in sort:
        rows = sorted(rows, key=lambda x: x[keys.index(k)])

    headers = [key_remap.get(key, key) for key in keys]
    print(tabulate.tabulate(rows, headers=headers,
                            floatfmt='.3f', tablefmt='simple',
                            numalign='center', stralign='center',
                            showindex=True, disable_numparse=True))

    if not ignored:
        return

    print('Ignored logs:')
    for log_dir in ignored:
        print('  - %s' % log_dir)
