# MLmidterm

## The Problem

What causes a passenger to be satisfied or unsatisfied with their flight? What features impact a passengers satisfaction on a flight?

## Data

There are 129880 total rows, and 22 columns describing the Passenger and the flight. Our target variable is Satisfaction.

* Gender: Gender of the passengers (Female, Male)
* Customer Type: The customer type (Loyal customer, disloyal customer)
* Age: The actual age of the passengers
* Type of Travel: Purpose of the flight of the passengers (Personal Travel, Business Travel)
* Class: Travel class in the plane of the passengers (Business, Eco, Eco Plus)
* Flight distance: The flight distance of this journey
* Inflight wifi service: Satisfaction level of the inflight wifi service (0:Not Applicable;1-5)
* Departure/Arrival time convenient: Satisfaction level of Departure/Arrival time convenient
* Ease of Online booking: Satisfaction level of online booking
* Gate location: Satisfaction level of Gate location
* Food and drink: Satisfaction level of Food and drink
* Online boarding: Satisfaction level of online boarding
* Seat comfort: Satisfaction level of Seat comfort
* Inflight entertainment: Satisfaction level of inflight entertainment
* On-board service: Satisfaction level of On-board service
* Leg room service: Satisfaction level of Leg room service
* Baggage handling: Satisfaction level of baggage handling
* Check-in service: Satisfaction level of Check-in service
* Inflight service: Satisfaction level of inflight service
* Cleanliness: Satisfaction level of Cleanliness
* Departure Delay in Minutes: Minutes delayed when departure
* Arrival Delay in Minutes: Minutes delayed when Arrival
* Satisfaction: Airline satisfaction level(Satisfaction, neutral or dissatisfaction)

## Classification

A passenger can either be satisfied or unsatisfied. I used classification models to test what features will impact a passenger being satisfied with their flight.

* Decision Trees
* Random Forest
* XGBoost

## Evaluating the model

I used AUC because this is a binary classification problem.

## Project Files

- ReadMe.md
- notebook.ipynb
  - Date prep and cleaning
  - EDA
  - Building the models and tuning parameters
- train.py
  - Train the final model and save it to .bin file
- xgb_model.bin
  - the Model and Dict Vectorizer
- predict.py
  - Loading the model and serving to a Flask Service at port 9696
- predict-test.py
  - Testing the flask service and the Model with example passengers
- Dockerfile
- Pipfile and Pipfile.lock

## Running the project

To see the full dataset, EDA, and model selection process, run **notebook.ipynb**

To run the model and save it to xgb_model.bin file:
`python train.py`

To test the model:
`python predict.py`
or
`gunicorn --bind 0.0.0.0:9696 predict:app`

In second terminal, run:
`python predict_test.py`

### Build the Docker Image

`docker build -t satisfaction .`

`docker run -it -p 9696:9696 satisfaction:latest `

Run in another terminal:

`python predict_test.py`
