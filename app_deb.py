import requests

new_image = {'filename':'example_02.png'}
#endpoint of the models
end_point ="http://127.0.0.1:5000/models/prediction"
#requests post method
resp = requests.post(end_point, json=new_image)
print(resp.text)










