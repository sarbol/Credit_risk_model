import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'UnsecuredLines':0.766127, 'age':45, 'PastDue30-59days':2, 'DebtRatio':0.802982, 'MonthlyIncome':9120, 'OpenCredit':13,
'Late_90days':0, 'RealEstate': 6, 'PastDue60-89days': 0, 'Dependents':2})

print(r.json())
