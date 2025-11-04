import pandas as pd
import time
import os
import datetime
import ccxt

exchange = ccxt.okx()

def crawl_exchanges_datas(exchange_name, symbol, start_time, end_time):
    """
    爬取交易所数据的方法.
    :param exchange_name:  交易所名称.
    :param symbol: 请求的symbol: like BTC/USDT, ETH/USD等。
    :param start_time: like 2018-1-1
    :param end_time: like 2019-1-1
    :return:
    """

    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class()
    print(exchange)

    current_path = os.getcwd()
    file_dir = os.path.join(current_path, exchange_name, symbol.replace('/', ''))

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)


    start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d')
    end_time = datetime.datetime.strptime(end_time, '%Y-%m-%d')

    start_time_stamp = int(time.mktime(start_time.timetuple())) * 1000
    end_time_stamp = int(time.mktime(end_time.timetuple())) * 1000

    print(start_time_stamp)  # 1529233920000
    print(end_time_stamp)

    while True:
        try:

            print(start_time_stamp)
            data = exchange.fetch_ohlcv(symbol, timeframe='1m', since=start_time_stamp)
            df = pd.DataFrame(data)
            df.rename(columns={0: 'open_time', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'volume'}, inplace=True)

            start_time_stamp = int(df.iloc[-1]['open_time'])  # 获取下一个次请求的时间.

            filename = str(start_time_stamp) + '.csv'
            save_file_path = os.path.join(file_dir, filename)

            print("文件保存路径为：%s" % save_file_path)
            df.set_index('open_time', drop=True, inplace=True)
            df.to_csv(save_file_path)

            if start_time_stamp > end_time_stamp:
                print("完成数据的请求.")
                break

            time.sleep(0.1)

        except Exception as error:
            print(error)
            time.sleep(1)

def sample_datas(exchange_name, symbol):
    
    path = os.path.join(os.getcwd(), exchange_name, symbol.replace('/', ''))
    print(path)
    file_paths = []
    for root, dirs, files in os.walk(path):
        if files:
            for file in files:
                if file.endswith('.csv'):
                    file_paths.append(os.path.join(path, file))

    file_paths = sorted(file_paths)
    df_lst = []

    for file in file_paths:
        df_lst.append(pd.read_csv(file))
    all_df = pd.concat(df_lst, axis=0)

    all_df = all_df.sort_values(by='open_time', ascending=True)

    #print(all_df)

    return all_df


def clear_datas(exchange_name, symbol):
    df = sample_datas(exchange_name, symbol)
    df['open_time'] = df['open_time'].apply(lambda x: (x//60)*60)
    df['Datetime'] = pd.to_datetime(df['open_time'], unit='ms')
    df.drop_duplicates(subset=['open_time'], inplace=True)
    df.set_index('Datetime', inplace=True)
    symbol_path = symbol.replace('/', '')
    df.to_csv(f'{exchange_name}_{symbol_path}_1min_data.csv')

#crawl_exchanges_datas('okx', 'BTC/USDT', '2022-01-01', '2025-01-01')
clear_datas('okx','BTC/USDT')