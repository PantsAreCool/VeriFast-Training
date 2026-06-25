# Task 3: Create a Chained Function Calling Pipeline
# Design a scenario where function calls must be chained: the output of one function becomes the input to another. 
# Example: (1) search_flights returns flight IDs, (2) get_flight_details takes a flight ID and returns pricing, 
# (3) calculate_total computes the final price with taxes. Implement this pipeline with proper error handling at each step, 
# and create a visualization of the function call chain showing data flow between steps.

import json

def search_flights(origin: str, destination: str) -> dict:
    if origin == "NYC" and destination == "LON":
        return {"status": "success", "flight_ids": ["FL-101", "FL-202"]}
    return {"status": "error", "message": "No flights found for this route."}

def get_flight_details(flight_id: str) -> dict:
    db = {
        "FL-101": {"base_price": 500, "class": "Economy"},
        "FL-202": {"base_price": 1200, "class": "Business"}
    }
    if flight_id in db:
        return {"status": "success", "details": db[flight_id]}
    return {"status": "error", "message": f"Flight ID {flight_id} not found."}

def calculate_total(base_price: float, tax_rate: float = 0.15) -> dict:
    if base_price <= 0:
        return {"status": "error", "message": "Invalid base price."}
    total = base_price * (1 + tax_rate)
    return {"status": "success", "total_price": round(total, 2)}

def run_flight_pipeline(origin: str, destination: str):
    print(f"Step 1: Searching flights from {origin} to {destination}...")
    res1 = search_flights(origin, destination)
    if res1["status"] == "error":
        return f"Pipeline Failed at Step 1: {res1['message']}"
    
    selected_id = res1["flight_ids"][0]
    
    print(f"Step 2: Fetching details for flight {selected_id}...")
    res2 = get_flight_details(selected_id)
    if res2["status"] == "error":
        return f"Pipeline Failed at Step 2: {res2['message']}"
        
    price = res2["details"]["base_price"]
    
    print(f"Step 3: Calculating final price for ${price}...")
    res3 = calculate_total(price)
    if res3["status"] == "error":
        return f"Pipeline Failed at Step 3: {res3['message']}"
        
    return f"Pipeline Success! Final Ticket Price: ${res3['total_price']}"



print(run_flight_pipeline("NYC", "LON"))