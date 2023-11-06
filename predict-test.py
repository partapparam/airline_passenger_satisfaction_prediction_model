import requests

url = "http://0.0.0.0:9696/predict"

survey_one = {"gender": 'male', 
            "customer_type": 'loyal customer', 
            "type_of_travel": 'business travel',
            "class": 'business',
            "age": 49,
            "flight_distance": 2000,
            "inflight_wifi_service": 4,
            "departure/arrival_time_convenient": 4,
            "ease_of_online_booking": 5, 
            "gate_location": 2, 
            "food_and_drink": 4,
            "online_boarding": 4,
            "seat_comfort": 4,
            "inflight_entertainment": 4,
            "leg_room_service": 2,
            "baggage_handling": 4,
            "checkin_service": 5, 
            "inflight_service": 4,
            "cleanliness": 3,
            "departure_delay_in_minutes": 25,
            "arrival_delay_in_minutes": 15,
            }

survey_two = {"gender": 'female', 
            "customer_type": 'disloyal Customer', 
            "type_of_travel": 'Personal Travel',
            "class": 'Eco',
            "age": 29,
            "flight_distance": 20,
            "inflight_wifi_service": 1,
            "departure/arrival_time_convenient": 1,
            "ease_of_online_booking": 1, 
            "gate_location": 1, 
            "food_and_drink": 1,
            "online_boarding": 1,
            "seat_comfort": 1,
            "inflight_entertainment": 1,
            "leg_room_service": 1,
            "baggage_handling": 1,
            "checkin_service": 1, 
            "inflight_service": 1,
            "cleanliness": 1,
            "departure_delay_in_minutes": 0,
            "arrival_delay_in_minutes": 0,
            }
response = requests.post(url, json=survey_two).json()


print(response)


