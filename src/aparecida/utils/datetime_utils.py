from datetime import UTC, datetime


def get_now_datetime_stamp():
    now: datetime = datetime.now(tz=UTC)
    return now.strftime("%y-%m-%d_%H-%M-%S")
