import requests

'''
invoke this from command line to test that render deployment works
'''


url = "https://census-fastapi-githubactions.onrender.com/"
# Make GET request
response = requests.get(url)

# Print status code and response
print(f"Status Code: {response.status_code}")
print("Response JSON:", response.json())



#Make post request
url = "https://census-fastapi-githubactions.onrender.com/predict"

payload = {
        "age": 52,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 209642,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 15024,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States"
    }

response = requests.post(url, json=payload)

print("Status Code:", response.status_code)

try:
    print("Response:", response.json())
except requests.exceptions.JSONDecodeError:
    print("Non-JSON response:")
    print(response.text)


