import pandas as pd
import requests
import os
from dotenv import load_dotenv
import datetime
import math
import time
from multiprocessing import Process

"""
    NASA API includes many RESTful requests each serving a different purpose
    Below I have used an API for nearest earth object
    It provides the dataset for all the asteroids (scanned) from 2015
    However, the API restricts the number of days to be less than or equal to 7 for which the request is sent
    Hence, I send multiple requests to the API each containing a different starting date
    Further, I have used multiple processes that run on multiple cores
    Lastly, make sure to generate api key and set up a .env file
"""

"""
    MANUALLY INSPECTED THE DATA AND CAME WITH THE LIST
    Required Variables (raw)
    id
    name
    absolute_magnitude_h
    estimated_diameter (json) --> kilometers (json) --> estimated_diameter_min and estimated_diameter_max
    is_poentially_hazardous_asteroid
    close_approach_data (list) (0th Index) --> relative_velocity (json) --> kilometers per hour
    close_approach_data (list) (0th Index) --> miss_distance (json) --> kilometers
    orbiting_body
    is_sentry_object
"""


########## Find number of cores in a system ############
CPU_CORES = os.cpu_count()
print("Available number of cores for processing: ", CPU_CORES)







########## Create custom date range with 8 days step so as to cover a week worth of data ############
start = datetime.datetime(2001, 1, 1)
end = datetime.datetime(2022, 6, 15)
step = datetime.timedelta(days=8)

result = []
while(start < end):
    result.append(start.strftime("%Y-%m-%d"))
    start += step


print("Required total requests: ", len(result))






######### Fetch API Key for NASA from environment file #############
load_dotenv()
API_KEY = os.getenv("NASA_API_KEY")  # Get the .env api key









######### Define a variable for storing all the necessary attributes from the API ############
required_features = [
    'id',
    'name',
    'absolute_magnitude',
    'est_diameter_min',
    'est_diameter_max',
    'relative_velocity',
    'miss_distance',
    'orbiting_body',
    'sentry_object',
    'hazardous'
]








######### Function for each process to follow ##########
def init(start, offset, res):
    ####### WRITE PID INTO TEXT FILE ############
    f = open("pid.txt", 'a')
    write_id = str(os.getpid()) + "\n"
    f.write(write_id)
    f.close()







    df = pd.DataFrame(columns=required_features)
    iterations = 0
    for start_date_index in range(start, start + offset):
        try:
            iterations += 1
            print("Iteration Number for pid ", os.getpid(), ": ", iterations)
            url = "https://api.nasa.gov/neo/rest/v2/feed?start_date=" + res[start_date_index] + "&api_key=" + API_KEY # format the url
            response = requests.get(url)  # "get" request on generated URL
            
            remaining_requests = int(response.headers.get("X-RateLimit-Remaining"))

            if(remaining_requests > 0):
                res_data = response.json()
                required_data = res_data.get("near_earth_objects")

                for date in required_data:
                    get_date_arr = required_data.get(date)
                    for data in get_date_arr:
                        temp_dict = {}  # empty dictionary
                        temp_dict['id'] = data.get("id")
                        temp_dict['name'] = data.get("name")
                        temp_dict['absolute_magnitude'] = data.get('absolute_magnitude_h')
                        temp_dict['est_diameter_min'] = data.get('estimated_diameter').get('kilometers').get('estimated_diameter_min')
                        temp_dict['est_diameter_max'] = data.get('estimated_diameter').get('kilometers').get('estimated_diameter_max')
                        temp_dict['relative_velocity'] = data.get("close_approach_data")[0].get('relative_velocity').get('kilometers_per_hour')
                        temp_dict['miss_distance'] = data.get("close_approach_data")[0].get('miss_distance').get('kilometers')
                        temp_dict['orbiting_body'] = data.get("close_approach_data")[0].get("orbiting_body")
                        temp_dict['sentry_object'] = data.get("is_sentry_object")
                        temp_dict['hazardous'] = data.get("is_potentially_hazardous_asteroid")
                        temp_df = pd.DataFrame(temp_dict, index=[0])
                        df = pd.concat([df, temp_df])
            else:
                time_to_sleep = 390
                if(len(response.headers.get('Retry-After')) > 0):
                    time_to_sleep = int(response.headers.get('Retry-After'))
                print("Process halted with id ", os.getpid(), " at ", datetime.datetime.now(), " for ", time_to_sleep, " with starting data index is: ", start_date_index)
                time.sleep(time_to_sleep)
        except:
            print("Due to some error, skipping date: ", result[start_date_index])
    df_name = "./Dataset/" + str(os.getpid()) + ".csv"
    df.to_csv(df_name, index=False)








####### Execute in main thread #########
if __name__ == "__main__":
    total_iterations = len(result)
    temp = math.floor(total_iterations / 7)

    process_arr = []

    for i in range(7):
        p = Process(target = init, args = (i*temp, temp, result, ))
        process_arr.append(p)
        p.start()
    
    for p in process_arr:
        p.join()
    








    ########## All Processes Completed ############
    print("Starting Dataset Merging")
    final_df = pd.DataFrame(columns = required_features)
    f = open('pid.txt', 'r')
    for i in range(7):
        pid = f.readline()
        pid = pid[:-1]
        path = './Dataset/' + pid + '.csv'
        temp_df = pd.read_csv(path)
        if os.path.exists(path):
            os.remove(path)
        final_df = pd.concat([final_df, temp_df])
    
    if os.path.exists('pid.txt'):
        os.remove('pid.txt')

    final_df.to_csv("./Dataset/neo.csv", index = False)