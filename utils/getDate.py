import datetime
from pytz import timezone
now = datetime.datetime.now()
fiveMinlater = now + datetime.timedelta(minutes=5)
stockholm = timezone('Asia/Kolkata')
print(fiveMinlater.astimezone(stockholm).strftime("%S %M %H %d %m ? %Y"))