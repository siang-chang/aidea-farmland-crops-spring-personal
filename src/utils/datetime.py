#!/usr/bin/env python
# coding: utf-8

# %%
from datetime import datetime
from PIL import Image
def get_taken_time(image_path):
    """get taken time from image file,  
    the input file must be the original image file output by the camera.

    Args:
        image_path (str): image file path.

    Returns:
        datetime: YYYY-MM-DD HH:MM:SS
    """
    image = Image.open(image_path)
    exif_data = image._getexif()
    return datetime.strptime(exif_data[36867], "%Y:%m:%d %H:%M:%S") if exif_data else None


def get_taken_month(image_path: str) -> int:
    """Get taken month from image file.

    Args:
        image_path (str): Image file path.

    Returns:
        int: The taken month in image file property.
    """
    month = get_taken_time(image_path).month
    return month

# %%
def transform_day_of_year(day_of_year):
    """transform day_of_year through winter solstice and summer solstice.
    
    summer solstice = 173, winter solstice = 356.

    Args:
        day_of_year(int): day of year. (1 to 365)

    Returns:
        int: day of year been transformed. (0.0 to 1.0)
    """

    if(day_of_year > 356):
        return (day_of_year-356)/((173+365)-356)
    elif(day_of_year < 173):
        return (day_of_year+365-356)/((173+365)-356)
    else:
        return 1-((day_of_year-173)/(356-173))


# %%
if __name__ == "__main__":
    
    import os
    read_from_dir = '../data/sample/'

    for file in os.listdir(read_from_dir):
        image_taken_time = get_taken_time(read_from_dir+file)
        print('image_taken_time:', image_taken_time)
        
        day_of_year = image_taken_time.timetuple().tm_yday
        print('day_of_year:', day_of_year)
        
        day_of_year_transformed = transform_day_of_year(day_of_year)
        print('day_of_year_transformed:', day_of_year_transformed)

# %%
