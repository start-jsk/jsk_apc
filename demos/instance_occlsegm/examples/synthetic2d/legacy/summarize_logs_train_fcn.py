#!/usr/bin/env python

import datetime
import json
import os
import os.path as osp
import yaml

import dateutil.parser
import pandas
import tabulate


def summarize_log(logs_dir, name, keys, target_key, objective,
                  show_range=True):
    try:
        params = yaml.load(open(osp.join(logs_dir, name, 'params.yaml')))
    except Exception:
        return None, osp.join(logs_dir, name)

    log_file = osp.join(logs_dir, name, 'log')
    try:
        df = pandas.DataFrame(json.load(open(log_file)))
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


def _summarize_log(args):
    return summarize_log(*args)


def summarize_logs(logs_dir, keys, target_key, objective, sort=None,
                   show_range=True, as_df=False):
    assert objective in ['min', 'max']
    assert target_key in keys

    args_list = []
    for name in os.listdir(logs_dir):
        args_list.append((
            logs_dir,
            name,
            keys,
            target_key,
            objective,
            show_range,
        ))

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = executor.map(_summarize_log, args_list)

    rows = []
    ignored = []
    for row, log_dir_ignored in results:
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
    print(tabulate.tabulate(rows, headers=keys,
                            floatfmt='.3f', tablefmt='simple',
                            numalign='center', stralign='center',
                            showindex=True, disable_numparse=True))

    if not ignored:
        return

    print('Ignored logs:')
    for log_dir in ignored:
        print('  - %s' % log_dir)


if __name__ == '__main__':
    here = osp.dirname(osp.abspath(__file__))
    logs_dir = osp.join(here, 'logs', 'train_fcn')
    print('# logs_dir = %s' % logs_dir)

    keys = [
        'name',
        'last_time',
        'dataset',
        'git_hash',
        'hostname',
        'model',
        'freeze',
        'epoch',
        'iteration',
        'validation/main/miou',
    ]
    summarize_logs(logs_dir, keys, target_key=keys[-1], objective='max')
