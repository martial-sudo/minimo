import datetime
class DayNightDetector:
        def is_night_time(self):
            """
            Check if current time is between 10 PM and 6 AM
            
            Returns:
            Boolean indicating if it's night time
            """
            current_hour = datetime.datetime.now().hour
            
            # Night time is between 10 PM (22:00) and 6 AM (6:00)
            return current_hour >= 22 or current_hour < 6