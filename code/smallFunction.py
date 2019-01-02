import time

def time2timestamp_ms(dt):
    # 转换成时间数组
    timeArray = time.strptime(dt, "%Y-%m-%d")
    # 转换成时间戳
    timestamp = time.mktime(timeArray)
    return str(int(timestamp)*1000)

if __name__ == '__main__':
    dt = "2018-10-31"
    print(time2timestamp_ms(dt))
