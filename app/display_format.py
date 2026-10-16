"""Human-readable values for inspection results and saved history."""
from datetime import datetime, timedelta, timezone


THAI_WEEKDAYS = ("จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์")
THAI_MONTHS = ("มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
               "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม")
BANGKOK = timezone(timedelta(hours=7))


def thai_detection_time(value: str) -> str:
    if not value:
        return ""
    moment = datetime.fromisoformat(value)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=BANGKOK)
    moment = moment.astimezone(BANGKOK)
    return (f"วัน{THAI_WEEKDAYS[moment.weekday()]}ที่ {moment.day} "
            f"{THAI_MONTHS[moment.month - 1]} {moment.year + 543} "
            f"เวลา {moment:%H:%M:%S} น.")
