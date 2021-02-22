import requests

# URL
url = 'http://localhost:5000/api/'

# Change the value of experience that you want to test
payload = {
	'UnsecuredLines':0.766,
	'PastDue30-59days':2,
	'Late_90days':0,
	'PastDue60-89days':0

}

r = requests.post(url,json=payload)

print(r.json())
